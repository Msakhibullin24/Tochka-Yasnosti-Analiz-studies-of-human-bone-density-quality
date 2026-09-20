"""HTTP API for batch DXA quality control (FastAPI). Local only: no outbound calls.

POST /api/v1/batch            multipart: one .zip or several DICOM files -> {"job_id": ...}; ?wait=true blocks
POST /api/v1/batch/path       json {"input_path": "/data/in"} for a folder/zip mounted into the container
GET  /api/v1/jobs/{id}        status + progress + summary
GET  /api/v1/jobs/{id}/results.csv | results.xlsx | additional_series.zip | rows
GET  /api/v1/jobs/{id}/overlay?path=<explanation_png>  PNG explanation
POST /api/v1/analyze          multipart: single DICOM -> JSON verdict
POST /api/v1/compare          multipart: baseline + followup DICOM (+ optional BMD, LSC) -> follow-up comparability
GET  /health
"""
from __future__ import annotations

import json
import os
import asyncio
from contextlib import asynccontextmanager
import fcntl
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .pipeline import Analyzer, Options, run_batch, write_failure_report
from .store import Store, Conflict
from .review import Review, Region, measurements
from .roi import evaluate_geometry

DATA_DIR = Path(os.environ.get("DXAQC_DATA_DIR", tempfile.gettempdir())) / "dxaqc-jobs"
ALLOWED_INPUT_ROOT = Path(os.environ.get("DXAQC_INPUT_ROOT", "/data")).resolve()
MAX_UPLOAD_BYTES = int(os.environ.get("DXAQC_MAX_UPLOAD_MB", "8192")) * 1024 * 1024

_analyzer: Analyzer | None = None
_lock = threading.Lock()


def store() -> Store:
    return Store(DATA_DIR)


@asynccontextmanager
async def lifespan(app):
    repository = store()
    # One worker process owns this local data directory, including crash recovery.
    with open(DATA_DIR / "worker.lock", "a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in repository.interrupted():
            exc = RuntimeError("processing interrupted by service restart; submit input again")
            write_failure_report(Path(job["input"]), Path(job["out"]), exc)
            repository.update(job["id"], status="failed", error=str(exc))
        stop = threading.Event()
        worker = threading.Thread(target=_worker, args=(repository, stop), daemon=True, name="dxaqc-worker")
        app.state.worker = worker
        worker.start()
        try:
            yield
        finally:
            stop.set()
            await asyncio.to_thread(worker.join)
            fcntl.flock(lease, fcntl.LOCK_UN)


app = FastAPI(title="Osseo AI - DXA quality control", version=__version__, lifespan=lifespan)

def analyzer() -> Analyzer:
    global _analyzer
    with _lock:
        if _analyzer is None:
            _analyzer = Analyzer()
    return _analyzer


def _worker(repository: Store, stop: threading.Event) -> None:
    while not stop.is_set():
        job = repository.claim()
        if job is None:
            stop.wait(0.2)
            continue
        job_id = job["id"]
        try:
            def progress(i, n):
                repository.update(job_id, done=i, total=n)
            from .source_archive import retain_input
            source = retain_input(Path(job["input"]), Path(job["out"]).parent)
            summary = run_batch(source, Path(job["out"]), Options(keep_explanation_dir=True),
                                analyzer(), progress, cancelled=lambda: repository.get(job_id)["cancel_requested"])
            repository.update(job_id, summary=summary, status="cancelled" if summary.get("cancelled") else "finished")
        except Exception as exc:
            try:
                write_failure_report(Path(job["input"]), Path(job["out"]), exc)
            except OSError:
                pass
            repository.update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if job.get("upload_dir"):
                shutil.rmtree(job["upload_dir"], ignore_errors=True)


def _submit(input_path: Path, upload_dir: Path | None = None) -> dict:
    job_id = uuid.uuid4().hex
    out = DATA_DIR / job_id / "out"
    try:
        out.mkdir(parents=True, exist_ok=True)
        return store().add(job_id, input_path, out, upload_dir)
    except OSError as exc:
        raise HTTPException(507, f"working folder is not writable: {exc}") from exc


def _public(job: dict) -> dict:
    return {**{k: job.get(k) for k in ("id", "status", "done", "total", "summary", "error", "created_at", "updated_at")},
            "originals_available": all((Path(job["out"]).parent / name).is_file() for name in ("source.zip", "source.sha256"))}


def _job(job_id: str) -> dict:
    job = store().get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


async def _wait(job_id: str) -> dict:
    while True:
        job = _job(job_id)
        if job["status"] not in ("queued", "running"):
            return job
        await asyncio.sleep(0.1)


@app.get("/api/v1/jobs")
def jobs(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return [_public(j) for j in store().list(limit, offset)]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "model_loaded": _analyzer is not None}


@app.get("/api/v1/ready")
def ready():
    try:
        worker = getattr(app.state, "worker", None)
        if worker is None or not worker.is_alive():
            raise RuntimeError("processing worker is not running")
        instance = analyzer()
        from .embedding import _model
        _model()
        repository = store()
        with tempfile.TemporaryFile(dir=DATA_DIR) as test:
            test.write(b"ready")
            test.flush()
        return {"ready": True, "version": __version__, "model": instance.bundle.meta,
                "disk_free_bytes": shutil.disk_usage(DATA_DIR).free,
                "jobs_in_recent_history": len(repository.list())}
    except Exception as exc:
        return JSONResponse({"ready": False, "error": str(exc)}, status_code=503)


@app.get("/api/v1/catalog")
def catalog():
    from .model import CRITERIA, VIOLATION_RU
    return {"criteria": VIOLATION_RU, "regions": {"spine": list(CRITERIA["spine"]),
            "hip_left": [*CRITERIA["hip"], "hip_metal_implant"],
            "hip_right": [*CRITERIA["hip"], "hip_metal_implant"]}}


@app.post("/api/v1/batch")
async def batch(files: list[UploadFile] = File(...), wait: bool = False):
    upload = DATA_DIR / ("upload-" + uuid.uuid4().hex)
    try:
        upload.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(507, f"working folder {DATA_DIR} is not writable: {exc}") from exc
    total = 0
    stored = []
    try:
        if len(files) > 1 and any((f.filename or "").lower().endswith(".zip") for f in files):
            raise HTTPException(422, "upload one ZIP archive or multiple DICOM files")
        for i, f in enumerate(files):
            name = Path(f.filename or f"file_{i}").name or f"file_{i}"
            folder = upload / f"{i:06d}"
            folder.mkdir()
            target = folder / name
            with open(target, "xb") as out:
                while chunk := await f.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "upload is too large")
                    out.write(chunk)
            stored.append(target)
        single_zip = len(stored) == 1 and stored[0].suffix.lower() == ".zip"
        job = _submit(stored[0] if single_zip else upload, upload)
    except Exception:
        shutil.rmtree(upload, ignore_errors=True)
        raise
    if wait:
        job = await _wait(job["id"])
    return _public(job)


class PathRequest(BaseModel):
    input_path: str


@app.post("/api/v1/batch/path")
async def batch_path(req: PathRequest, wait: bool = False):
    p = Path(req.input_path).resolve()
    if ALLOWED_INPUT_ROOT not in p.parents and p != ALLOWED_INPUT_ROOT:
        raise HTTPException(400, f"input_path must be inside {ALLOWED_INPUT_ROOT}")
    if not p.exists():
        raise HTTPException(404, "input_path does not exist")
    job = _submit(p)
    if wait:
        job = await _wait(job["id"])
    return _public(job)


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str):
    return _public(_job(job_id))


@app.get("/api/v1/jobs/{job_id}/source.zip")
def source_download(job_id: str):
    path = Path(_job(job_id)["out"]).parent / "source.zip"
    if not path.is_file():
        raise HTTPException(404, "original DICOM was not retained for this job")
    return FileResponse(path, filename="source.zip")


@app.post("/api/v1/jobs/{job_id}/reprocess")
def reprocess(job_id: str):
    job = _job(job_id)
    if job["status"] in ("running", "queued"):
        raise HTTPException(409, "wait for processing to finish")
    source = Path(job["out"]).parent / "source.zip"
    if not source.is_file():
        raise HTTPException(409, "original DICOM unavailable; upload the original files again")
    try:
        expected = (source.parent / 'source.sha256').read_text().strip()
    except OSError:
        raise HTTPException(409, 'original archive integrity record unavailable')
    import hashlib
    with source.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise HTTPException(409, "original archive integrity check failed")
    new = _submit(source)
    (Path(new['out']).parent / 'lineage.json').write_text(json.dumps({'parent_job_id': job_id, 'source_sha256': expected}))
    return _public(new)


@app.post("/api/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    _job(job_id)
    if not store().cancel(job_id):
        raise HTTPException(409, "job has already finished")
    return {"cancel_requested": True}


def _result_rows(job_id: str):
    import csv
    path = Path(_job(job_id)["out"]) / "results.csv"
    if not path.exists():
        raise HTTPException(409, "results are not ready")
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@app.get("/api/v1/jobs/{job_id}/rows")
def job_rows(job_id: str):
    return _result_rows(job_id)


@app.get("/api/v1/jobs/{job_id}/worklist")
def job_worklist(job_id: str):
    """Queue projection; official CSV and machine rows remain immutable."""
    rows = _result_rows(job_id)
    latest = store().latest_reviews(job_id)
    for row in rows:
        review = latest.get(row.get("row_id"))
        row["review_status"] = (review["status"] if review else "unreviewed") if row["processing_status"] == "Success" else "unavailable"
        row["review_revision"] = str(review["revision"] if review else 0)
        actions = review.get("followups", []) if review else []
        row["open_actions"] = str(sum(a["state"] == "open" for a in actions))
        row["second_opinion_requested"] = "1" if any(a["state"] == "open" and a["kind"] == "second_opinion" for a in actions) else "0"
    return rows


@app.get("/api/v1/jobs/{job_id}/overlay")
def overlay(job_id: str, path: str):
    """PNG explanation; `path` is the value of the `explanation_png` column."""
    base = (Path(_job(job_id)["out"]) / "additional_series").resolve()
    target = (base / path).resolve()
    if base not in target.parents or target.suffix != ".png" or not target.exists():
        raise HTTPException(404, "overlay not found")
    return FileResponse(target)


def _image_path(job_id: str, row_id: str, suffix: str) -> Path:
    if len(row_id) != 24 or any(c not in "0123456789abcdef" for c in row_id):
        raise HTTPException(404, "image not found")
    path = Path(_job(job_id)["out"]) / "images" / (row_id + suffix)
    if not path.is_file():
        raise HTTPException(404, "image not found")
    return path


@app.get("/api/v1/jobs/{job_id}/images/{row_id}")
def image_detail(job_id: str, row_id: str):
    return json.loads(_image_path(job_id, row_id, ".json").read_text(encoding="utf-8"))


class GeometryEvaluation(BaseModel):
    geometry: list[Region] = Field(max_length=40)


@app.post("/api/v1/jobs/{job_id}/images/{row_id}/geometry-evaluation")
def geometry_evaluation(job_id: str, row_id: str, body: GeometryEvaluation):
    return evaluate_geometry(body.geometry, image_detail(job_id, row_id))


@app.get("/api/v1/jobs/{job_id}/images/{row_id}/original.png")
def original_image(job_id: str, row_id: str):
    return FileResponse(_image_path(job_id, row_id, ".png"))


@app.get("/api/v1/jobs/{job_id}/images/{row_id}/reviews")
def image_reviews(job_id: str, row_id: str):
    _image_path(job_id, row_id, ".json")
    return store().reviews(job_id, row_id)


@app.post("/api/v1/jobs/{job_id}/images/{row_id}/reviews")
def save_review(job_id: str, row_id: str, review: Review):
    if _job(job_id)["status"] not in ("finished", "cancelled"):
        raise HTTPException(409, "wait for processing to finish")
    detail = image_detail(job_id, row_id)
    try:
        review.for_region(detail["region"])
        document = review.model_dump(exclude={"expected_revision"})
        document["measurements"] = measurements(review.geometry, detail["width"], detail["height"],
                                                detail["pixel_mm_x"], detail["pixel_mm_y"])
        document["measurement_scale_source"] = detail["pixel_mm_source"]
        document["roi_evaluation"] = evaluate_geometry(review.geometry, detail)
        return store().review(job_id, row_id, review.expected_revision, document)
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/v1/jobs/{job_id}/reviews.json")
def export_reviews(job_id: str):
    import csv
    job = _job(job_id)
    if job["status"] not in ("finished", "cancelled"):
        raise HTTPException(409, "wait for processing to finish")
    with open(Path(job["out"]) / "results.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return JSONResponse({"schema_version": "1", "job_id": job_id, "source": "expert_reviews",
                         "images": [{"machine_result": r, "reviews": store().reviews(job_id, r["row_id"])}
                                    for r in rows if r.get("row_id")]},
                        headers={"Content-Disposition": 'attachment; filename="reviews.json"'})


@app.get("/api/v1/jobs/{job_id}/review-package.zip")
def review_package(job_id: str):
    """Export one snapshot of review revisions; never relabel machine SC/SR as expert output."""
    import io
    import zipfile
    import hashlib
    from .review_report import render_review
    snapshot = json.loads(export_reviews(job_id).body)
    payloads = {"reviews.json": json.dumps(snapshot, ensure_ascii=False, indent=2).encode()}
    manifest = {"job_id": job_id, "source": "expert_review_snapshot", "images": [],
                "limitations": "No expert DICOM SC/SR generated. Drafts and unreviewed images are not final conclusions."}
    for image in snapshot['images']:
        row, history = image['machine_result'], image['reviews']
        last = history[-1] if history else None
        entry = {"row_id": row['row_id'], "image_uid": row['image_uid'],
                 "status": last['status'] if last else 'unreviewed',
                 "revision": last['revision'] if last else None}
        if last and last['status'] != 'draft':
            detail = image_detail(job_id, row['row_id'])
            # render_review signature is shared with the individual revision endpoint.
            payloads[f"reviews/{row['row_id']}-v{last['revision']}.html"] = render_review(row, detail, last, _image_path(job_id, row["row_id"], ".png").read_bytes()).encode()
        manifest['images'].append(entry)
    manifest['complete'] = all(i['status'] in ('confirmed', 'not_evaluable') for i in manifest['images'])
    manifest['sha256'] = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    payloads['manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    return Response(buffer.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename="review-package.zip"'})


@app.get("/api/v1/jobs/{job_id}/reviewed.csv")
def reviewed_csv(job_id: str):
    import csv
    import io
    from .model import official_violation_type
    job = _job(job_id)
    if job["status"] not in ("finished", "cancelled"):
        raise HTTPException(409, "wait for processing to finish")
    with open(Path(job["out"]) / "results.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["row_id", "image_uid", "path_to_file", "review_revision",
                            "review_status", "review_author", "review_quality_class", "review_violation_type", "comment"])
    writer.writeheader()
    for row in rows:
        history = store().reviews(job_id, row.get("row_id", ""))
        if not history:
            continue
        last = history[-1]
        # A new draft supersedes the previous confirmation; do not export it as final.
        if last["status"] == "draft":
            continue
        writer.writerow({"row_id": row.get("row_id"), "image_uid": row["image_uid"],
                         "path_to_file": row["path_to_file"], "review_revision": last["revision"],
                         "review_status": last["status"], "review_author": last["author"],
                         "review_quality_class": last["quality_class"],
                         "review_violation_type": official_violation_type(last["violations"]), "comment": last["comment"]})
    return Response("\ufeff" + output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="reviewed.csv"'})


@app.get("/api/v1/jobs/{job_id}/report.html", response_class=HTMLResponse)
def printable_report(job_id: str):
    import csv
    from html import escape
    from .result_context import model_verdict, assessment_notes
    from .model import official_violation_type
    job = _job(job_id)
    if job["status"] in {"queued", "running"}:
        raise HTTPException(409, "wait for processing to finish")
    with open(Path(job["out"]) / "results.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    items = []
    for row in rows:
        revisions = store().reviews(job_id, row.get("row_id", ""))
        last = revisions[-1] if revisions else None
        human = "Не проверено специалистом"
        if last:
            verdict = ("Невозможно оценить" if last["quality_class"] is None else
                       official_violation_type(last["violations"]) if last["quality_class"] else "Нарушений нет")
            status_label = {"draft": "Черновик", "confirmed": "Подтверждено", "not_evaluable": "Неоценимо"}[last["status"]]
            human = f"{verdict} · {last['author']} · {status_label} · версия {last['revision']} · {last['comment']}"
        items.append("<tr>" + "".join("<td>" + escape(str(v)) + "</td>" for v in (
            row["path_to_file"], row["anatomical_region"],
            (model_verdict(row) + " · " + " ".join(assessment_notes(row))) if row["processing_status"] == "Success" else row["error_message"],
            human)) + "</tr>")
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Osseo AI — отчёт проверки</title>'
            '<style>body{font:16px system-ui;margin:32px;color:#14213a}table{border-collapse:collapse;width:100%}'
            'td,th{border:1px solid #bbb;padding:10px;text-align:left;overflow-wrap:anywhere}'
            '@media print{body{margin:0}tr{break-inside:avoid}}</style>'
            '<h1>Отчёт контроля качества DXA</h1><p>Вывод ИИ и экспертная проверка представлены отдельно. '
            'Черновик не является подтверждённым решением.</p><table><thead><tr><th>Изображение</th>'
            '<th>Область</th><th>Результат модели</th><th>Проверка специалиста</th></tr></thead><tbody>'
            + ''.join(items) + '</tbody></table></html>')


@app.get("/api/v1/jobs/{job_id}/images/{row_id}/reviews/{revision}/report.html", response_class=HTMLResponse)
def revision_report(job_id: str, row_id: str, revision: int):
    from .review_report import render_review
    detail = image_detail(job_id, row_id)
    review = next((r for r in store().reviews(job_id, row_id) if r["revision"] == revision), None)
    row = next((r for r in _result_rows(job_id) if r.get("row_id") == row_id), None)
    if review is None or row is None:
        raise HTTPException(404, "review revision not found")
    return HTMLResponse(render_review(row, detail, review, _image_path(job_id, row_id, ".png").read_bytes()),
                        headers={"Content-Disposition": f'attachment; filename="review-{row_id}-v{revision}.html"'})


@app.get("/api/v1/jobs/{job_id}/study-report.html", response_class=HTMLResponse)
def study_report(job_id: str, study_uid: str):
    from html import escape
    from .review_report import render_review
    if not study_uid:
        raise HTTPException(422, "study UID is required")
    rows = [r for r in _result_rows(job_id) if r.get("study_uid") == study_uid]
    if not rows:
        raise HTTPException(404, "study not found")
    latest = store().latest_reviews(job_id)
    parts = []
    for row in rows:
        row_id = row.get("row_id")
        review = latest.get(row_id)
        if review and row["processing_status"] == "Success":
            parts.append(render_review(row, image_detail(job_id, row_id), review, _image_path(job_id, row_id, ".png").read_bytes(), fragment=True))
        else:
            message = row.get("error_message", "") if row["processing_status"] != "Success" else "Не проверено специалистом"
            parts.append(f'<h2>{escape(row["path_to_file"])}</h2><p>{escape(message)}</p><p>Результат модели: {escape(row.get("violation_type") or ("Нарушений не выявлено" if row["processing_status"] == "Success" else "Недоступен"))}</p>')
    return HTMLResponse('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка исследования</title>'
                        '<style>body{font:16px system-ui;max-width:1000px;margin:24px auto;padding:16px}svg{display:block;max-width:100%;max-height:650px;background:#10151c}article{break-before:page}p,li{white-space:pre-wrap;overflow-wrap:anywhere}@media print{svg{max-height:180mm}}</style>'
                        f'<h1>Карточка проверки исследования</h1><p>StudyInstanceUID: {escape(study_uid)}</p><p>Изображений: {len(rows)}. Каждая проверка содержит собственный номер версии. Непроверенные и ошибочные изображения включены отдельно.</p>'
                        + ''.join('<article>' + p + '</article>' for p in parts) + '</html>',
                        headers={"Content-Disposition": 'attachment; filename="study-review.html"'})


@app.get("/api/v1/jobs/{job_id}/validation.json")
def validation_report(job_id: str):
    from .validate_results import validate
    job = _job(job_id)
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "results are not ready")
    return JSONResponse(validate(Path(job["out"]) / "results.csv"),
                        headers={"Content-Disposition": 'attachment; filename="validation.json"'})


@app.get("/api/v1/jobs/{job_id}/{name}")
def job_file(job_id: str, name: str):
    if name not in {"results.csv", "results.xlsx", "results_extended.csv", "submission.csv", "submission.xlsx", "submission_validation.json", "additional_series.zip", "summary.json"}:
        raise HTTPException(404, "unknown artefact")
    path = Path(_job(job_id)["out"]) / name
    if not path.exists():
        raise HTTPException(409, "artefact is not ready or was not produced")
    return FileResponse(path, filename=name)


@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...)):
    from .dicom_io import DicomReadError, read_dxa
    payload = await file.read(512 * 1024 * 1024 + 1)
    if len(payload) > 512 * 1024 * 1024:
        raise HTTPException(413, "file is too large")
    with tempfile.NamedTemporaryFile(suffix=".dcm") as tmp:
        tmp.write(payload)
        tmp.flush()
        try:
            img = read_dxa(tmp.name)
            res = await asyncio.to_thread(lambda: analyzer().analyze(img))
            from .anatomy import image_assessment
            res["assessment"] = await asyncio.to_thread(image_assessment, img, res)
        except DicomReadError as exc:
            return JSONResponse({"processing_status": "Failure", "error_code": exc.code, "error_message": str(exc)}, 422)
    res.pop("overlay", None)
    return json.loads(json.dumps({"processing_status": "Success", "study_uid": img.study_uid,
                                  "image_uid": img.image_uid, **res}, default=float).replace("NaN", "null"))


@app.post("/api/v1/compare")
async def compare_visits(baseline: UploadFile = File(...), followup: UploadFile = File(...),
                         baseline_bmd: float | None = Form(None), followup_bmd: float | None = Form(None),
                         lsc_percent: float | None = Form(None), precision_cv_percent: float | None = Form(None)):
    """Dynamics: are two visits comparable by positioning, and is the (operator-supplied) BMD change significant?"""
    from .dicom_io import DicomReadError, read_dxa
    from .dynamics import compare
    results = []
    for upload in (baseline, followup):
        payload = await upload.read(512 * 1024 * 1024 + 1)
        if len(payload) > 512 * 1024 * 1024:
            raise HTTPException(413, "file is too large")
        with tempfile.NamedTemporaryFile(suffix=".dcm") as tmp:
            tmp.write(payload)
            tmp.flush()
            try:
                results.append(await asyncio.to_thread(lambda: analyzer().analyze(read_dxa(tmp.name))))
            except DicomReadError as exc:
                return JSONResponse({"processing_status": "Failure", "error_code": exc.code,
                                     "error_message": f"{upload.filename}: {exc}"}, 422)
    out = compare(results[0], results[1], baseline_bmd=baseline_bmd, followup_bmd=followup_bmd,
                  lsc_percent=lsc_percent, precision_cv_percent=precision_cv_percent).to_dict()
    return json.loads(json.dumps({"processing_status": "Success", **out}, default=float).replace("NaN", "null"))


UI_ROOT = Path(__file__).parent / "static" / "app"
if (UI_ROOT / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=UI_ROOT / "assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    target = UI_ROOT / "index.html"
    if not target.exists():
        raise HTTPException(503, "UI not built; run npm run build and copy dist into competition/dxaqc/static/app")
    return target.read_text(encoding="utf-8")
