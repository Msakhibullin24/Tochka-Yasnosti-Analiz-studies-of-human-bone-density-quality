"""Verify public dataset downloads and record archive inventories without unpacking.

Publisher hashes take precedence. For sources without published hashes, the
local SHA-256 is an integrity fingerprint, not proof of publisher authenticity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend/scripts"))
from download_training_data import matches  # noqa: E402


def archive_inventory(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            name = PurePosixPath(member.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Symlink in archive: {member.filename}")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Archive CRC failure: {bad}")
        files = [m for m in members if not m.is_dir()]
        return {
            "crc_verified": True,
            "files": len(files),
            "uncompressed_bytes": sum(m.file_size for m in files),
            "extensions": dict(Counter(PurePosixPath(m.filename).suffix.lower() for m in files)),
            "members": [m.filename for m in files],
        }


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(root: Path, cache: dict) -> dict:
    catalog = json.loads((ROOT / "docs/competition/download_sources.json").read_text())
    acquisition = json.loads((ROOT / "docs/competition/additional_dataset_acquisition.json").read_text())
    items = []
    for dataset, source in catalog.items():
        for spec in source["files"]:
            items.append({**spec, "dataset": dataset, "source": source["source"],
                          "license": source["license"], "digest_origin": "Publisher metadata or pinned source catalog"})
    items.extend(acquisition["files"])
    rows = []
    for spec in items:
        relative = f"{spec['dataset']}/{spec['path']}"
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe path: {relative}")
        row = {**spec, "relative_path": relative, "status": "not_downloaded"}
        partial = path.with_name(path.name + ".part")
        candidate = path if path.is_file() else partial
        if candidate.is_file():
            row["local_bytes"] = candidate.stat().st_size
            # Managed downloaders own their .part -> final transaction, including
            # the final CRC check. Do not race a parallel archive downloader by
            # adopting a full-sized .part while it is still validating it.
            managed_lock = path.with_name(path.name + '.download.lock')
            if candidate == partial and (spec not in acquisition["files"] or managed_lock.exists()):
                row["status"] = "downloading"
            elif candidate.stat().st_size != spec["bytes"]:
                row["status"] = "downloading" if candidate == partial else "size_mismatch"
            else:
                signature = [candidate.stat().st_size, candidate.stat().st_mtime_ns]
                saved = cache.get(relative, {})
                try:
                    if saved.get("signature") == signature:
                        details = saved["details"]
                    else:
                        if "digest" in spec and not matches(candidate, spec):
                            raise ValueError("Publisher/pinned checksum mismatch")
                        details = {"sha256": fingerprint(candidate)}
                        if spec["path"].lower().endswith(".zip"):
                            details["archive"] = archive_inventory(candidate)
                    if candidate == partial:
                        if path.exists():
                            raise ValueError("Refusing to overwrite existing archive")
                        candidate.rename(path)
                        signature = [path.stat().st_size, path.stat().st_mtime_ns]
                    cache[relative] = {"signature": signature, "details": details}
                    row.update(details)
                    row["status"] = "verified"
                except (OSError, ValueError, zipfile.BadZipFile) as exc:
                    row["status"] = "invalid"
                    row["error"] = str(exc)
        elif any(path.parent.glob(".download-*")):
            row["status"] = "pending_or_downloading"
        rows.append(row)
    return {"schema_version": 1, "generated_unix": time.time(),
            "clinical_validation_complete": False,
            "files": rows, "unavailable": acquisition["unavailable"],
            "statuses": dict(Counter(r["status"] for r in rows)),
            "verified_bytes": sum(r["bytes"] for r in rows if r["status"] == "verified")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/external")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args()
    report_path = args.root / "dataset-acquisition-audit.json"
    cache_path = args.root / ".dataset-audit-cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    deadline = time.monotonic() + args.hours * 3600
    while True:
        report = audit(args.root, cache)
        for path, value in [(report_path, report), (cache_path, cache)]:
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            temporary.replace(path)
        print(json.dumps({"statuses": report["statuses"], "verified_bytes": report["verified_bytes"]}), flush=True)
        if not args.watch or all(r["status"] == "verified" for r in report["files"]) or time.monotonic() >= deadline:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
