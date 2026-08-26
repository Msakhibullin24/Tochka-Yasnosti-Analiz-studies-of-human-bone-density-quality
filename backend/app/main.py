from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import ModelUnavailable
from .preprocessing import InvalidDicom, prepare_dicom
from .routing import InvalidProtocolOverride, registry_snapshot
from .service import UnsupportedProtocol, analyze
from .settings import settings
from third_party.dxa_pointplacement import UPSTREAM_COMMIT


app = FastAPI(
    title="Osseo AI DXA inference",
    version="0.1.0",
    description="Research inference service integrating hawaii-ai/dxa-pointplacement.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "checkpointAvailable": settings.checkpoint_path.is_file(),
        "upstreamCommit": UPSTREAM_COMMIT,
    }


@app.get("/api/v1/models")
def models() -> dict[str, object]:
    return {"models": registry_snapshot()}


@app.post("/api/v1/studies/analyze")
async def analyze_study(
    file: UploadFile = File(...),
    protocol_override: str | None = Form(default=None),
) -> dict[str, object]:
    filename = file.filename or "study.dcm"
    if not filename.lower().endswith((".dcm", ".dicom")) and file.content_type not in ("application/dicom", "application/dicom+json"):
        raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_MEDIA_TYPE", "message": "A DICOM file is required"})
    payload = await file.read(settings.max_upload_bytes + 1)
    if not payload:
        raise HTTPException(status_code=400, detail={"code": "EMPTY_FILE", "message": "The DICOM file is empty"})
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail={"code": "FILE_TOO_LARGE", "message": "The DICOM file exceeds the upload limit"})
    try:
        return analyze(prepare_dicom(payload), protocol_override)
    except InvalidDicom as error:
        raise HTTPException(status_code=400, detail={"code": "INVALID_DICOM", "message": str(error)}) from error
    except UnsupportedProtocol as error:
        raise HTTPException(status_code=422, detail={"code": "UNSUPPORTED_PROTOCOL", "message": str(error), "routing": error.decision.as_dict()}) from error
    except InvalidProtocolOverride as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_PROTOCOL_OVERRIDE", "message": str(error)}) from error
    except ModelUnavailable as error:
        raise HTTPException(status_code=503, detail={"code": "MODEL_UNAVAILABLE", "message": str(error)}) from error
