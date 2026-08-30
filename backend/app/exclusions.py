from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_write_lock = threading.Lock()
MAX_LOG_BYTES = 16 * 1024 * 1024


def record_exclusion(path: Path, payload: bytes, reason: str) -> dict[str, Any]:
    """Persist a PHI-free audit entry for a rejected presentation object."""
    entry: dict[str, Any] = {
        "objectId": f"EX-{hashlib.sha256(payload).hexdigest()[:20].upper()}",
        "reason": reason,
        "trainingEligible": False,
        "byteSize": len(payload),
        "recordedAt": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with _write_lock:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)
    return entry


def read_exclusions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    if path.stat().st_size > MAX_LOG_BYTES:
        raise ValueError("Exclusion registry exceeds the safety limit")
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                values.append(json.loads(line))
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in reversed(values):
        object_id = str(value.get("objectId", ""))
        if object_id and object_id not in seen:
            seen.add(object_id)
            unique.append(value)
    return unique
