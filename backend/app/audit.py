from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_lock = threading.Lock()
MAX_AUDIT_BYTES = 64 * 1024 * 1024
GENESIS_HASH = "0" * 64


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def pseudonymous_actor(value: str, key: str = "") -> str:
    if not key:
        return "ACT-UNKEYED-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12].upper()
    return "ACT-" + hmac.new(key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()[:16].upper()


def read_audit(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    if path.stat().st_size > MAX_AUDIT_BYTES:
        raise ValueError("Audit trail exceeds the safety limit")
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid audit JSON at line {line_number}") from error
    return values


def verify_audit(path: Path) -> dict[str, Any]:
    values = read_audit(path)
    previous_hash = GENESIS_HASH
    for index, value in enumerate(values, 1):
        event_hash = str(value.get("eventHash", ""))
        content = {key: item for key, item in value.items() if key != "eventHash"}
        expected = hashlib.sha256(_canonical(content)).hexdigest()
        if value.get("sequence") != index or value.get("previousHash") != previous_hash or event_hash != expected:
            return {"valid": False, "eventCount": len(values), "firstInvalidSequence": index, "headHash": previous_hash}
        previous_hash = event_hash
    return {"valid": True, "eventCount": len(values), "firstInvalidSequence": None, "headHash": previous_hash}


def append_audit(
    path: Path,
    *,
    action: str,
    object_id: str,
    request_id: str,
    actor_id: str = "SYSTEM",
    outcome: str = "success",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a PHI-free, hash-chained domain event."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _lock:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            existing = read_audit(path)
            sequence = len(existing) + 1
            previous_hash = str(existing[-1]["eventHash"]) if existing else GENESIS_HASH
            content: dict[str, Any] = {
                "schemaVersion": "1.0.0",
                "sequence": sequence,
                "eventId": f"AE-{sequence:010d}",
                "recordedAt": datetime.now(timezone.utc).isoformat(),
                "requestId": request_id,
                "actorId": actor_id,
                "action": action,
                "objectId": object_id,
                "outcome": outcome,
                "details": details or {},
                "previousHash": previous_hash,
            }
            event = {**content, "eventHash": hashlib.sha256(_canonical(content)).hexdigest()}
            encoded = _canonical(event) + b"\n"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        return event
