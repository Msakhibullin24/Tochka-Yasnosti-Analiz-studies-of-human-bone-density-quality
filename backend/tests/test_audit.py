from __future__ import annotations

import json
from pathlib import Path

from app.audit import append_audit, pseudonymous_actor, verify_audit


def test_actor_pseudonym_requires_key_for_production_grade_identifier() -> None:
    assert pseudonymous_actor("reader-01").startswith("ACT-UNKEYED-")
    keyed = pseudonymous_actor("reader-01", "separate-audit-secret")
    assert keyed.startswith("ACT-")
    assert "reader-01" not in keyed
    assert keyed != pseudonymous_actor("reader-01", "different-audit-secret")


def test_hash_chained_audit_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    first = append_audit(
        path,
        action="annotation_saved",
        object_id="ST-0123456789ABCDEFFEDC",
        request_id="request-0001",
        actor_id=pseudonymous_actor("reader-01"),
        details={"readIndex": 1},
    )
    second = append_audit(
        path,
        action="annotations_exported",
        object_id="CURRENT_DATASET",
        request_id="request-0002",
    )
    assert first["eventHash"] == second["previousHash"]
    assert verify_audit(path) == {
        "valid": True,
        "eventCount": 2,
        "firstInvalidSequence": None,
        "headHash": second["eventHash"],
    }
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    values[0]["outcome"] = "tampered"
    path.write_text("\n".join(json.dumps(item) for item in values) + "\n", encoding="utf-8")
    result = verify_audit(path)
    assert result["valid"] is False
    assert result["firstInvalidSequence"] == 1
