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
import queue
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import __version__
from .pipeline import Analyzer, Options, run_batch

DATA_DIR = Path(os.environ.get("DXAQC_DATA_DIR", tempfile.gettempdir())) / "dxaqc-jobs"
ALLOWED_INPUT_ROOT = Path(os.environ.get("DXAQC_INPUT_ROOT", "/data")).resolve()
MAX_UPLOAD_BYTES = int(os.environ.get("DXAQC_MAX_UPLOAD_MB", "8192")) * 1024 * 1024

app = FastAPI(title="Osseo AI - DXA quality control", version=__version__)
_jobs: dict[str, dict] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_analyzer: Analyzer | None = None
_lock = threading.Lock()


def analyzer() -> Analyzer:
    global _analyzer
    with _lock:
        if _analyzer is None:
            _analyzer = Analyzer()
    return _analyzer


def _worker() -> None:
    while True:
        job_id = _queue.get()
        job = _jobs[job_id]
        job["status"] = "running"
        try:
            def progress(i, n):
                job["done"], job["total"] = i, n
            job["summary"] = run_batch(Path(job["input"]), Path(job["out"]), Options(keep_explanation_dir=True),
                                       analyzer(), progress)
            job["status"] = "finished"
        except BaseException as exc:  # job-level failure is reported, the worker thread must never die
            job["status"], job["error"] = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            job["event"].set()
            upload = job.get("upload_dir")
            if upload:
                shutil.rmtree(upload, ignore_errors=True)


threading.Thread(target=_worker, daemon=True, name="dxaqc-worker").start()


MAX_KEPT_JOBS = 50


def _prune() -> None:
    finished = [j for j in _jobs.values() if j["status"] in ("finished", "failed")]
    for old in finished[:-MAX_KEPT_JOBS]:
        shutil.rmtree(Path(old["out"]).parent, ignore_errors=True)
        _jobs.pop(old["id"], None)


def _submit(input_path: Path, upload_dir: Path | None = None) -> dict:
    _prune()
    job_id = uuid.uuid4().hex
    out = DATA_DIR / job_id / "out"
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(507, f"working folder {DATA_DIR} is not writable: {exc}") from exc
    _jobs[job_id] = {"id": job_id, "status": "queued", "input": str(input_path), "out": str(out), "done": 0,
                     "total": 0, "event": threading.Event(), "upload_dir": str(upload_dir) if upload_dir else None}
    _queue.put(job_id)
    return _jobs[job_id]


def _public(job: dict) -> dict:
    return {k: job.get(k) for k in ("id", "status", "done", "total", "summary", "error")}


def _job(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "job not found")
    return _jobs[job_id]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "model_loaded": _analyzer is not None}


@app.post("/api/v1/batch")
async def batch(files: list[UploadFile] = File(...), wait: bool = False):
    upload = DATA_DIR / ("upload-" + uuid.uuid4().hex)
    try:
        upload.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(507, f"working folder {DATA_DIR} is not writable: {exc}") from exc
    total = 0
    for i, f in enumerate(files):
        name = Path(f.filename or f"file_{i}").name or f"file_{i}"
        target = upload / name
        if target.exists():  # same file name uploaded twice: keep both
            target = upload / f"{i:05d}_{name}"
        with open(target, "wb") as out:
            while chunk := await f.read(1 << 20):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    shutil.rmtree(upload, ignore_errors=True)
                    raise HTTPException(413, "upload is too large")
                out.write(chunk)
    stored = sorted(upload.iterdir())
    single_zip = len(stored) == 1 and stored[0].suffix.lower() == ".zip"
    job = _submit(stored[0] if single_zip else upload, upload)
    if wait:
        job["event"].wait()
    return _public(job)


class PathRequest(BaseModel):
    input_path: str


@app.post("/api/v1/batch/path")
def batch_path(req: PathRequest, wait: bool = False):
    p = Path(req.input_path).resolve()
    if ALLOWED_INPUT_ROOT not in p.parents and p != ALLOWED_INPUT_ROOT:
        raise HTTPException(400, f"input_path must be inside {ALLOWED_INPUT_ROOT}")
    if not p.exists():
        raise HTTPException(404, "input_path does not exist")
    job = _submit(p)
    if wait:
        job["event"].wait()
    return _public(job)


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str):
    return _public(_job(job_id))


@app.get("/api/v1/jobs/{job_id}/rows")
def job_rows(job_id: str):
    import csv
    path = Path(_job(job_id)["out"]) / "results.csv"
    if not path.exists():
        raise HTTPException(409, "results are not ready")
    with open(path, encoding="utf-8-sig", newline="") as f:
        return JSONResponse(list(csv.DictReader(f)))


@app.get("/api/v1/jobs/{job_id}/overlay")
def overlay(job_id: str, path: str):
    """PNG explanation; `path` is the value of the `explanation_png` column."""
    base = (Path(_job(job_id)["out"]) / "additional_series").resolve()
    target = (base / path).resolve()
    if base not in target.parents or target.suffix != ".png" or not target.exists():
        raise HTTPException(404, "overlay not found")
    return FileResponse(target)


@app.get("/api/v1/jobs/{job_id}/{name}")
def job_file(job_id: str, name: str):
    if name not in {"results.csv", "results.xlsx", "additional_series.zip", "summary.json"}:
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
            res = analyzer().analyze(img)
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
                results.append(analyzer().analyze(read_dxa(tmp.name)))
            except DicomReadError as exc:
                return JSONResponse({"processing_status": "Failure", "error_code": exc.code,
                                     "error_message": f"{upload.filename}: {exc}"}, 422)
    out = compare(results[0], results[1], baseline_bmd=baseline_bmd, followup_bmd=followup_bmd,
                  lsc_percent=lsc_percent, precision_cv_percent=precision_cv_percent).to_dict()
    return json.loads(json.dumps({"processing_status": "Success", **out}, default=float).replace("NaN", "null"))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
