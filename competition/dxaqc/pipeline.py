"""Batch processing: archive/folder of DICOM studies -> result table (+ explanation series).

Guarantees required by the task statement:
  * no unhandled exceptions - every file ends up as a row with Success or Failure;
  * reproducible - fixed seeds, CPU inference, sorted traversal, content-derived UIDs;
  * local only - no network access anywhere in this package.
"""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import time
import traceback
import zipfile
import resource
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .dicom_io import DicomReadError, DxaImage, looks_like_dicom, read_dxa, read_identifiers
from .model import LATERALITY, REGION_LABEL, VIOLATION_RU, QualityBundle, group_of, official_violation_type
from .report import write_csv, write_xlsx
from .decision import decide

SKIP_SUFFIXES = {".xlsx", ".xls", ".csv", ".txt", ".json", ".xml", ".png", ".jpg", ".jpeg", ".pdf", ".md",
                 ".html", ".db", ".ini", ".sha256", ".log"}
SKIP_NAMES = {"dicomdir", "thumbs.db", ".ds_store"}
MAX_ARCHIVE_BYTES = 20 * 1024 ** 3
MAX_ARCHIVE_FILES = 200_000
LOW_REGION_CONFIDENCE = 0.6  # below this the image may be a region the service does not support
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "bundle.joblib"


class OutputConflictError(ValueError):
    """Refuse to overwrite a previous run or write into the input tree."""


def check_output(input_path: Path, out_dir: Path):
    source, output = input_path.resolve(), out_dir.resolve()
    if source == output or (input_path.is_dir() and source in output.parents):
        raise OutputConflictError("output must be outside the input directory")
    reserved = ('submission.csv', 'submission.xlsx', 'submission_validation.json', 'results_extended.csv', 'results.csv', 'results.xlsx', 'summary.json', 'additional_series',
                'additional_series.zip', 'images', '_extracted')
    if any((out_dir / name).exists() or (out_dir / name).is_symlink() for name in reserved):
        raise OutputConflictError("output contains a previous run; choose a new output directory")


@dataclass
class Options:
    explanations: bool = True  # overlay PNG + DICOM Secondary Capture + DICOM SR
    keep_explanation_dir: bool = False  # the API keeps the folder to serve PNGs; the CLI keeps only the zip
    strict_columns: bool = False
    xlsx: bool = True


def load_bundle(path: Path = MODEL_PATH) -> QualityBundle:
    import joblib
    from .dicom_io import CALIBRATION_VERSION
    bundle = joblib.load(path)
    if bundle.meta.get("calibration_version") != CALIBRATION_VERSION:
        raise ValueError("model calibration mismatch: retrain the bundle for organiser V2")
    return bundle


def failure_row(path: str, code: str, message: str) -> dict:
    return {"path_to_study": path, "path_to_file": path, "study_uid": "", "image_uid": "",
            "quality_class": 1, "violation_type": "", "processing_status": "Failure",
            "time_of_processing": 0.0, "error_code": code, "error_message": message[:300]}


def write_tables(rows: list[dict], out_dir: Path, opts: Options) -> None:
    write_csv(rows, out_dir / "results.csv", opts.strict_columns)
    write_csv(rows, out_dir / "results_extended.csv")
    write_csv(rows, out_dir / "submission.csv", True)
    if opts.xlsx:
        write_xlsx(rows, out_dir / "results.xlsx", opts.strict_columns)
        write_xlsx(rows, out_dir / "submission.xlsx", True)
    from .validate_results import validate
    acceptance = validate(out_dir / "submission.csv", competition=True)
    (out_dir / "submission_validation.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2))


def write_failure_report(input_path: Path, out_dir: Path, exc: Exception,
                         opts: Options | None = None) -> None:
    opts = opts or Options()
    out_dir.mkdir(parents=True, exist_ok=True)
    message = f"{type(exc).__name__}: {exc}"
    rows = [failure_row(str(input_path), "BATCH_INPUT_ERROR", message)]
    write_tables(rows, out_dir, opts)
    (out_dir / "summary.json").write_text(json.dumps({"error": message, "files": 1,
        "success": 0, "failure": 1}, ensure_ascii=False), encoding="utf-8")


def safe_extract(archive: Path, dest: Path, budget: dict | None = None) -> list[dict]:
    """Bounded extraction; return every rejected member as a report row.

    A shared budget counts all nested archives, not just each ZIP separately.
    Existing paths are never overwritten by colliding archive members.
    """
    budget = budget if budget is not None else {"files": 0, "bytes": 0}
    failures = []
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        budget["files"] += len(infos)
        budget["bytes"] += sum(i.file_size for i in infos)
        if budget["files"] > MAX_ARCHIVE_FILES or budget["bytes"] > MAX_ARCHIVE_BYTES:
            raise ValueError("archive is too large")
        root = dest.resolve()
        for info in infos:
            name = info.filename
            if not info.flag_bits & 0x800:
                try:
                    recoded = name.encode("cp437").decode("cp866")
                    if any("\u0400" <= ch <= "\u04ff" for ch in recoded):
                        name = recoded
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            target = (dest / name).resolve()
            if root not in target.parents or (info.external_attr >> 16) & 0o170000 == 0o120000:
                failures.append(failure_row(name, "UNSAFE_ARCHIVE_MEMBER", "unsafe path or symbolic link"))
                continue
            if info.is_dir():
                continue
            created = False
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "xb") as out:
                    created = True
                    shutil.copyfileobj(src, out)
            except (OSError, zipfile.BadZipFile, RuntimeError, EOFError) as exc:
                if created:
                    target.unlink(missing_ok=True)
                failures.append(failure_row(name, "ARCHIVE_MEMBER_ERROR", str(exc)))
    return failures


def discover(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*"), key=lambda q: str(q)):
        if not p.is_file() or p.name.lower() in SKIP_NAMES or p.name.startswith("._"):
            continue
        if p.suffix.lower() in SKIP_SUFFIXES or "__MACOSX" in p.parts:
            continue
        if not looks_like_dicom(p):  # e.g. .nrrd masks, office files: not an image of the study
            continue
        files.append(p)
    return files


def _round(d: dict, nd: int = 2) -> dict:
    return {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)) for k, v in d.items()}


class Analyzer:
    def __init__(self, bundle: QualityBundle | None = None):
        import torch

        torch.manual_seed(0)
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        self.bundle = bundle or load_bundle()

    def analyze(self, img: DxaImage) -> dict:
        from .embedding import embed
        from .geometry import measure_image

        raw_embedding = embed(img.pixels)[None]
        regions, conf = self.bundle.router.predict(raw_embedding)
        region, confidence = regions[0], float(conf[0])
        if not np.isfinite(confidence) or confidence < LOW_REGION_CONFIDENCE:
            raise DicomReadError("UNCERTAIN_REGION", "anatomical region is uncertain; automatic quality assessment withheld")
        px = np.ascontiguousarray(img.pixels[:, ::-1]) if region == "hip_left" else img.pixels
        meas = measure_image(px, region, img.pixel_mm, img.pixel_mm_x)
        gm = self.bundle.groups[group_of(region)]
        score, crit = gm.predict([meas.features], embed(px)[None] if region == "hip_left" else raw_embedding)
        score = float(score[0])
        crit = {k: float(v[0]) for k, v in crit.items()}
        decision = decide(group_of(region), score, crit, meas.features,
                          gm.quality_threshold, gm.criterion_thresholds)
        return {"region": region, "region_confidence": confidence, **decision,
                "criteria": crit, "features": meas.features, "overlay": meas.overlay}


def _row_base(rel: Path) -> dict:
    return {"path_to_study": rel.as_posix(), "path_to_file": rel.as_posix()}


def _safe(name: str, fallback: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in "._-")[:120].strip(".")
    return cleaned or fallback


def _common(parts_list: list[tuple]) -> tuple:
    out = []
    for level in zip(*parts_list):
        if len(set(level)) != 1:
            break
        out.append(level[0])
    return tuple(out)


def assign_study_paths(rows: list[dict]) -> None:
    """path_to_study = <wrapper folders>/<top folder of the study>, relative to the input root.

    The study folder is found from StudyInstanceUID grouping, so wrapper folders of an archive
    (e.g. 'Исследования/') are not mistaken for a study and deep series folders are not reported
    instead of the study. Files lying directly in the root (flat layout) keep their own path.
    """
    by_study: dict[str, list[dict]] = {}
    for r in rows:
        by_study.setdefault(r["study_uid"] or "\0" + str(Path(r["path_to_file"]).parent), []).append(r)
    deepest = {k: _common([Path(r["path_to_file"]).parent.parts for r in g]) for k, g in by_study.items()}
    wrapper = _common(list(deepest.values())) if len(deepest) > 1 else ()
    for key, group in by_study.items():
        d = deepest[key]
        study_dir = d[: len(wrapper) + 1] if len(d) > len(wrapper) else ()
        for r in group:
            r["path_to_study"] = "/".join(study_dir) if study_dir else r["path_to_file"]


def process_files(files: list[Path], root: Path, out_dir: Path, analyzer: Analyzer, opts: Options,
                  progress=None, cancelled=None) -> list[dict]:
    rows: list[dict] = []
    cache: dict[tuple, tuple[dict, str]] = {}
    extra = out_dir / "additional_series"
    from .source_roi import presentation_states, merge_presentation_roi
    files, presentations = presentation_states(files)
    for n, path in enumerate(files):
        if cancelled and cancelled():
            rows.extend(failure_row(p.relative_to(root).as_posix(), "CANCELLED", "cancelled before processing")
                        for p in files[n:])
            break
        t0 = time.perf_counter()
        rel = path.relative_to(root)
        row_id = hashlib.sha256(rel.as_posix().encode()).hexdigest()[:24]
        row = {"row_id": row_id, **_row_base(rel), "study_uid": "", "image_uid": "", "anatomical_region": "", "quality_class": 1,
               "violation_type": "", "processing_status": "Failure"}
        try:
            img = read_dxa(path)
            img.source_roi = merge_presentation_roi(img, presentations.get(img.image_uid, []))
            row["study_uid"], row["image_uid"] = img.study_uid, img.image_uid
            cache_key = (img.pixel_sha256, img.pixel_mm, img.pixel_mm_x)
            if cache_key in cache:
                res, first = cache[cache_key]
                row["duplicate_of"] = first
            else:
                res = analyzer.analyze(img)
            from .anatomy import image_assessment, display_overlay
            assessment = image_assessment(img, res)
            projection, source_roi = assessment['projection'], assessment['source_roi']
            row.update({
                "projection_assessment": json.dumps(projection, ensure_ascii=False),
                "anatomy_assessment": json.dumps(res['anatomy_candidates'], ensure_ascii=False),
                "source_roi_assessment": json.dumps(source_roi, ensure_ascii=False),
                "anatomical_checks_complete": "false",
                "criterion_states": json.dumps(res["criterion_states"], ensure_ascii=False),
                "violation_type_status": res["violation_type_status"],
                "review_reasons": json.dumps(res["review_reasons"], ensure_ascii=False),
                "image_width": img.pixels.shape[1],
                "image_height": img.pixels.shape[0],
                "anatomical_region": REGION_LABEL[res["region"]],
                "quality_class": res["quality"],
                "violation_type": official_violation_type(res["violations"]),
                "processing_status": "Success",
                "quality_prob": round(min(max(res["score"], 0.0), 1.0), 4),
                "laterality": LATERALITY[res["region"]],
                "violation_codes": ";".join(res["violations"]),
                "violation_scores": json.dumps(_round(res["criteria"], 3), ensure_ascii=False),
                "criterion_thresholds": json.dumps(res["criterion_thresholds"]),
                "decision_reason": res["decision_reason"],
                "decision_version": res["decision_version"],
                "violation_description": "; ".join(VIOLATION_RU.get(v, v) for v in res["violations"]),
                "measurements": json.dumps(_round(res["features"]), ensure_ascii=False),
                "region_confidence": round(res["region_confidence"], 4),
                "pixel_mm": round(img.pixel_mm, 4),
                "pixel_mm_x": round(img.pixel_mm_x or img.pixel_mm, 4),
                "pixel_mm_source": img.pixel_mm_source,
                "error_message": ";".join(img.warnings),
            })
            if opts.keep_explanation_dir:
                import cv2
                assets = out_dir / "images"
                assets.mkdir(exist_ok=True)
                if not cv2.imwrite(str(assets / f"{row_id}.png"), img.pixels):
                    raise OSError("cannot write original preview")
                geometry = display_overlay(res, img.pixels.shape[1])
                if res["region"] == "hip_left":
                    geometry = {k: [(img.pixels.shape[1] - 1 - x, y) for x, y in pts]
                                for k, pts in geometry.items()}
                detail = {"row_id": row_id, "region": res["region"], "width": img.pixels.shape[1],
                          "height": img.pixels.shape[0], "pixel_mm_x": img.pixel_mm_x or img.pixel_mm,
                          "pixel_mm_y": img.pixel_mm, "pixel_mm_source": img.pixel_mm_source,
                          "geometry": geometry, "model": analyzer.bundle.meta, "assessment": assessment}
                (assets / f"{row_id}.json").write_text(json.dumps(detail, ensure_ascii=False), encoding="utf-8")
            if opts.explanations:
                _write_explanations(img, res, row, extra)
            if "duplicate_of" not in row:
                cache[cache_key] = (res, img.image_uid)
        except DicomReadError as exc:
            row["study_uid"], row["image_uid"] = read_identifiers(path)
            row.update({"violation_codes": "processing_failure", "error_code": exc.code, "error_message": str(exc)[:300]})
        except Exception as exc:  # never let one file stop the batch
            row["study_uid"], row["image_uid"] = read_identifiers(path)
            row.update({"violation_codes": "processing_failure", "error_code": "INTERNAL_ERROR",
                        "error_message": f"{type(exc).__name__}: {exc}"[:300]})
            try:
                with open(out_dir / "errors.log", "a", encoding="utf-8") as log:
                    log.write(f"{rel}\n{traceback.format_exc()}\n")
            except OSError:
                pass
        row["time_of_processing"] = round(time.perf_counter() - t0, 4)
        rows.append(row)
        if progress:
            progress(n + 1, len(files))
    return rows


def _write_explanations(img: DxaImage, res: dict, row: dict, extra: Path) -> None:
    import cv2
    from .explain import render_overlay, write_secondary_capture, write_sr
    from .anatomy import display_overlay

    try:
        # Named by UIDs: file names such as CR000000.dcm repeat across series and studies.
        sub = _safe(img.study_uid, "study-" + img.pixel_sha256[:12])
        stem = _safe(img.image_uid, "image-" + img.pixel_sha256[:12])
        folder = extra / sub
        folder.mkdir(parents=True, exist_ok=True)
        row["explanation_png"] = f"{sub}/{stem}_qc.png"
        bgr = render_overlay(img.pixels, res["region"], display_overlay(res, img.pixels.shape[1]), res["features"], res["violations"],
                             res["quality"], res["score"], img.pixel_mm, img.pixel_mm_x, row=row)
        cv2.imwrite(str(folder / f"{stem}_qc.png"), bgr)
        write_secondary_capture(bgr, img.study_uid, img.image_uid, folder / f"{stem}_qc_sc.dcm", img.sop_class_uid)
        write_sr(row, img.study_uid, img.image_uid, img.sop_class_uid, folder / f"{stem}_qc_sr.dcm", img.series_uid)
    except Exception as exc:  # explanations are optional; the verdict row must survive
        row["error_message"] = (row.get("error_message", "") + f";EXPLANATION_FAILED:{type(exc).__name__}").strip(";")


def run_batch(input_path: Path, out_dir: Path, opts: Options | None = None, analyzer: Analyzer | None = None,
              progress=None, cancelled=None) -> dict:
    """Process a folder or .zip. Returns a summary; writes results.csv / results.xlsx / additional_series.zip."""
    opts = opts or Options()
    check_output(input_path, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    work = None
    archive_rows = []
    budget = {"files": 0, "bytes": 0}
    try:
        if input_path.is_file() and zipfile.is_zipfile(input_path):
            work = out_dir / "_extracted"
            archive_rows.extend(safe_extract(input_path, work, budget))
            for depth in range(4):
                inner_archives = [p for p in sorted(work.rglob("*"))
                                  if p.is_file() and p.suffix.lower() == ".zip"]
                if not inner_archives:
                    break
                for inner in inner_archives:
                    rel = inner.relative_to(work).as_posix()
                    try:
                        if depth == 3:
                            raise ValueError("nested archive depth exceeded")
                        if not zipfile.is_zipfile(inner):
                            raise ValueError("invalid nested ZIP")
                        issues = safe_extract(inner, inner.parent / (inner.name + ".extracted"), budget)
                        for issue in issues:
                            issue["path_to_file"] = issue["path_to_study"] = rel + "!/" + issue["path_to_file"]
                        archive_rows.extend(issues)
                    except Exception as exc:
                        archive_rows.append(failure_row(rel, "ARCHIVE_INPUT_ERROR", str(exc)))
                    finally:
                        inner.unlink(missing_ok=True)
            root = work
        elif input_path.is_dir():
            root = input_path
        elif input_path.is_file():
            root = input_path.parent
        else:
            raise FileNotFoundError(str(input_path))
        files = [input_path] if (input_path.is_file() and work is None) else discover(root)
        analyzer = analyzer or Analyzer()
        rows = process_files(files, root, out_dir, analyzer, opts, progress, cancelled)
        assign_study_paths(rows)
        rows.extend(archive_rows)
        if not rows:
            rows = [failure_row(str(input_path), "NO_DICOM_IMAGES", "input contains no DICOM images")]
    except Exception as exc:
        write_failure_report(input_path, out_dir, exc, opts)
        raise
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
    write_tables(rows, out_dir, opts)
    extra = out_dir / "additional_series"
    if extra.exists():
        with zipfile.ZipFile(out_dir / "additional_series.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(extra.rglob("*")):
                if p.is_file():
                    info = zipfile.ZipInfo(p.relative_to(extra).as_posix(), date_time=(2020, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(info, p.read_bytes())
        if not opts.keep_explanation_dir:
            shutil.rmtree(extra, ignore_errors=True)
    ok = sum(r["processing_status"] == "Success" for r in rows)
    times = [r["time_of_processing"] for r in rows] or [0.0]
    by_study: dict[str, float] = {}
    for r in rows:
        key = r["study_uid"] or r["path_to_study"]
        by_study[key] = by_study.get(key, 0.0) + r["time_of_processing"]
    summary = {"version": __version__, "submission_available": True,
               "submission_valid": json.loads((out_dir / "submission_validation.json").read_text())["valid"], "files": len(rows), "success": ok, "failure": len(rows) - ok,
               "success_rate": ok / max(len(rows), 1), "violations": sum(r["quality_class"] == 1 and r["processing_status"] == "Success" for r in rows),
               "cancelled": any(r.get("error_code") == "CANCELLED" for r in rows),
               "studies": len(by_study), "max_seconds_per_study": round(max(by_study.values(), default=0.0), 3),
               "mean_seconds_per_image": round(float(np.mean(times)), 4),
               "total_seconds": round(time.perf_counter() - t0, 3)}
    summary['process_peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
    summary['cpu_count_visible'] = os.cpu_count()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
