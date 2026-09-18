"""Batch processing: archive/folder of DICOM studies -> result table (+ explanation series).

Guarantees required by the task statement:
  * no unhandled exceptions - every file ends up as a row with Success or Failure;
  * reproducible - fixed seeds, CPU inference, sorted traversal, content-derived UIDs;
  * local only - no network access anywhere in this package.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .dicom_io import DicomReadError, DxaImage, looks_like_dicom, read_dxa, read_identifiers
from .model import LATERALITY, REGION_LABEL, VIOLATION_RU, QualityBundle, group_of, official_violation_type
from .report import write_csv, write_xlsx

SKIP_SUFFIXES = {".xlsx", ".xls", ".csv", ".txt", ".json", ".xml", ".png", ".jpg", ".jpeg", ".pdf", ".md",
                 ".html", ".db", ".ini", ".sha256", ".log"}
SKIP_NAMES = {"dicomdir", "thumbs.db", ".ds_store"}
MAX_ARCHIVE_BYTES = 20 * 1024 ** 3
MAX_ARCHIVE_FILES = 200_000
LOW_REGION_CONFIDENCE = 0.6  # below this the image may be a region the service does not support
IMPLANT_SAT_FRAC = 0.015  # normal hips <= 0.011, implants >= 0.018 on the organiser data
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "bundle.joblib"


@dataclass
class Options:
    explanations: bool = True  # overlay PNG + DICOM Secondary Capture + DICOM SR
    keep_explanation_dir: bool = False  # the API keeps the folder to serve PNGs; the CLI keeps only the zip
    strict_columns: bool = False
    xlsx: bool = True


def load_bundle(path: Path = MODEL_PATH) -> QualityBundle:
    import joblib
    return joblib.load(path)


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip without path traversal, symlinks or decompression bombs; nested zips one level deep."""
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if len(infos) > MAX_ARCHIVE_FILES or sum(i.file_size for i in infos) > MAX_ARCHIVE_BYTES:
            raise ValueError("archive is too large")
        root = dest.resolve()
        for info in infos:
            name = info.filename
            # Archives made on Windows/macOS frequently carry cp866/cp437 mojibake in Cyrillic names.
            if not info.flag_bits & 0x800:
                try:
                    recoded = name.encode("cp437").decode("cp866")
                    if any("\u0400" <= ch <= "\u04ff" for ch in recoded):
                        name = recoded
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            target = (dest / name).resolve()
            if root not in target.parents:
                continue
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
            except (OSError, zipfile.BadZipFile, RuntimeError, EOFError):
                continue  # corrupt / encrypted / colliding member: skip it, keep the rest of the batch


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
        from .geometry import measure_hip, measure_spine

        regions, conf = self.bundle.router.predict(embed(img.pixels)[None])
        region, confidence = regions[0], float(conf[0])
        px = np.ascontiguousarray(img.pixels[:, ::-1]) if region == "hip_left" else img.pixels
        meas = (measure_spine if region == "spine" else measure_hip)(px, img.pixel_mm)
        gm = self.bundle.groups[group_of(region)]
        score, crit = gm.predict([meas.features], embed(px)[None])
        score = float(score[0])
        crit = {k: float(v[0]) for k, v in crit.items()}
        quality = int(score >= gm.quality_threshold)
        violations = [k for k, p in crit.items() if p >= gm.criterion_thresholds.get(k, 0.5)] if quality else []
        if quality and not violations:
            violations = [max(crit, key=crit.get)]
        # Deterministic rule: a metal implant makes the hip unusable regardless of the learned score.
        if region != "spine" and meas.features.get("sat_frac", 0.0) >= IMPLANT_SAT_FRAC:
            quality, score = 1, max(score, 0.99)
            violations = sorted(set(violations) | {"hip_metal_implant"})
        return {"region": region, "region_confidence": confidence, "score": score, "quality": quality,
                "violations": violations, "criteria": crit, "features": meas.features, "overlay": meas.overlay}


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
                  progress=None) -> list[dict]:
    rows: list[dict] = []
    cache: dict[str, tuple[dict, str, str]] = {}
    extra = out_dir / "additional_series"
    for n, path in enumerate(files):
        t0 = time.perf_counter()
        rel = path.relative_to(root)
        row = {**_row_base(rel), "study_uid": "", "image_uid": "", "anatomical_region": "", "quality_class": 1,
               "violation_type": "", "processing_status": "Failure"}
        try:
            img = read_dxa(path)
            row["study_uid"], row["image_uid"] = img.study_uid, img.image_uid
            if img.pixel_sha256 in cache:  # exact pixel copy inside the batch: identical verdict by construction
                res, first, first_png = cache[img.pixel_sha256]
                row["duplicate_of"] = first
                row["explanation_png"] = first_png  # identical pixels -> the same visual explanation applies
            else:
                res = analyzer.analyze(img)
                if res["region_confidence"] < LOW_REGION_CONFIDENCE:
                    img.warnings.append("LOW_REGION_CONFIDENCE")
            row.update({
                "anatomical_region": REGION_LABEL[res["region"]],
                "quality_class": res["quality"],
                "violation_type": official_violation_type(res["violations"]),
                "processing_status": "Success",
                "quality_prob": round(min(max(res["score"], 0.0), 1.0), 4),
                "laterality": LATERALITY[res["region"]],
                "violation_codes": ";".join(res["violations"]),
                "violation_scores": json.dumps(_round(res["criteria"], 3), ensure_ascii=False),
                "violation_description": "; ".join(VIOLATION_RU.get(v, v) for v in res["violations"]),
                "measurements": json.dumps(_round(res["features"]), ensure_ascii=False),
                "region_confidence": round(res["region_confidence"], 4),
                "pixel_mm": round(img.pixel_mm, 4),
                "pixel_mm_source": img.pixel_mm_source,
                "error_message": ";".join(img.warnings),
            })
            if opts.explanations and "duplicate_of" not in row:
                _write_explanations(img, res, row, extra)
            if "duplicate_of" not in row:
                cache[img.pixel_sha256] = (res, img.image_uid, row.get("explanation_png", ""))
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

    try:
        # Named by UIDs: file names such as CR000000.dcm repeat across series and studies.
        sub = _safe(img.study_uid, "study-" + img.pixel_sha256[:12])
        stem = _safe(img.image_uid, "image-" + img.pixel_sha256[:12])
        folder = extra / sub
        folder.mkdir(parents=True, exist_ok=True)
        row["explanation_png"] = f"{sub}/{stem}_qc.png"
        bgr = render_overlay(img.pixels, res["region"], res["overlay"], res["features"], res["violations"],
                             res["quality"], res["score"], img.pixel_mm)
        cv2.imwrite(str(folder / f"{stem}_qc.png"), bgr)
        write_secondary_capture(bgr, img.study_uid, img.image_uid, folder / f"{stem}_qc_sc.dcm")
        write_sr(row, img.study_uid, img.image_uid, "1.2.840.10008.5.1.4.1.1.1", folder / f"{stem}_qc_sr.dcm")
    except Exception as exc:  # explanations are optional; the verdict row must survive
        row["error_message"] = (row.get("error_message", "") + f";EXPLANATION_FAILED:{type(exc).__name__}").strip(";")


def run_batch(input_path: Path, out_dir: Path, opts: Options | None = None, analyzer: Analyzer | None = None,
              progress=None) -> dict:
    """Process a folder or .zip. Returns a summary; writes results.csv / results.xlsx / additional_series.zip."""
    opts = opts or Options()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    work = None
    try:
        if input_path.is_file() and zipfile.is_zipfile(input_path):
            work = out_dir / "_extracted"
            safe_extract(input_path, work)
            for depth in range(3):  # nested archives, up to three levels
                # Only real .zip files: .xlsx/.docx are zip containers too and must not be unpacked.
                inner_archives = [p for p in sorted(work.rglob("*"))
                                  if p.is_file() and p.suffix.lower() == ".zip" and zipfile.is_zipfile(p)]
                if not inner_archives:
                    break
                for inner in inner_archives:
                    try:
                        safe_extract(inner, inner.parent / (inner.name + ".extracted"))
                        inner.unlink()
                    except Exception:
                        pass  # left in place: it becomes a Failure row instead of killing the batch
            root = work
        elif input_path.is_dir():
            root = input_path
        elif input_path.is_file():
            root = input_path.parent
        else:
            raise FileNotFoundError(str(input_path))
        files = [input_path] if (input_path.is_file() and work is None) else discover(root)
        analyzer = analyzer or Analyzer()
        rows = process_files(files, root, out_dir, analyzer, opts, progress)
        assign_study_paths(rows)
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
    write_csv(rows, out_dir / "results.csv", opts.strict_columns)
    if opts.xlsx:
        write_xlsx(rows, out_dir / "results.xlsx", opts.strict_columns)
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
    summary = {"version": __version__, "files": len(rows), "success": ok, "failure": len(rows) - ok,
               "success_rate": ok / max(len(rows), 1), "violations": sum(r["quality_class"] == 1 for r in rows),
               "studies": len(by_study), "max_seconds_per_study": round(max(by_study.values(), default=0.0), 3),
               "mean_seconds_per_image": round(float(np.mean(times)), 4),
               "total_seconds": round(time.perf_counter() - t0, 3)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
