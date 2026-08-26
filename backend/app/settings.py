from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    checkpoint_path: Path = Path(
        os.getenv("DXA_CHECKPOINT_PATH", BACKEND_ROOT / "models" / "checkpoint.pth")
    )
    device: str = os.getenv("DXA_DEVICE", "auto")
    allowed_origins: tuple[str, ...] = tuple(
        value.strip()
        for value in os.getenv("DXA_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
        if value.strip()
    )
    max_upload_bytes: int = int(os.getenv("DXA_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
    model_version: str = os.getenv("DXA_MODEL_VERSION", "hawaii-ai-dxa-points-7ac19eb")
    criteria_version: str = os.getenv("DXA_CRITERIA_VERSION", "DXA-TOTAL-BODY-QC-2026.1")


settings = Settings()
