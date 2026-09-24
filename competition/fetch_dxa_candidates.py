"""Fetch pinned research DXA candidates and retain compact inference weights.

DAX upstream files include training state. After SHA-256 verification, this tool
extracts teacher backbones to safetensors and deletes the large training files.
The DAX model cards license weights CC-BY-NC-4.0; never bundle them by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend/scripts"))
from download_training_data import download, matches  # noqa: E402
from fetch_specialists import extract_source  # noqa: E402

CATALOG = ROOT / "docs/competition/dxa_candidate_sources.json"


def spec(entry: dict) -> dict:
    return {**entry, "algorithm": "sha256", "digest": entry["sha256"]}


def verified(path: Path, entry: dict) -> bool:
    return matches(path, spec(entry))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_dax(source: Path, destination: Path, name: str, expected_sha256: str) -> None:
    # Official PyTorch checkpoint contains numpy scalar metadata; torch 2.1 cannot
    # safely unpickle it with weights_only=True. Never reach this call until the
    # complete official checkpoint has matched its pinned SHA-256.
    import torch
    from safetensors.torch import save_file

    if sha256_file(source) != expected_sha256:
        raise ValueError("Upstream DAX checkpoint checksum mismatch")
    teacher = torch.load(source, map_location="cpu", weights_only=False)["teacher"]
    prefix = "module.backbone." if name == "dax_resnet18_a" else "backbone."
    state = {key.removeprefix(prefix): value.detach().contiguous()
             for key, value in teacher.items() if key.startswith(prefix)}
    if len(state) < 100:
        raise ValueError(f"DAX teacher backbone has too few tensors: {len(state)}")
    temporary = destination.with_suffix(".safetensors.part")
    save_file(state, temporary)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check local assets without network or torch")
    parser.add_argument("--output", type=Path, default=ROOT / "data/specialists")
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    statuses = {}
    for source in catalog["sources"]:
        target = output / source["path"]
        destination = output / "sources" / source["id"]
        if not args.verify:
            download(output, {}, spec(source))
            extract_source(target, destination, source["sha256"])
        ok = verified(target, source) and (destination / ".osseo-source.json").is_file()
        statuses[source["id"]] = "verified" if ok else "missing_or_corrupt"
        print(source["id"], statuses[source["id"]], flush=True)
    for name, weight in catalog["weights"].items():
        target = output / weight["path"]
        if not args.verify and not verified(target, weight):
            if target.exists():
                raise ValueError(f"Existing candidate weight differs: {target}")
            if name == "lite_medsam":
                download(output, {}, spec(weight))
            else:
                upstream = output / "weights" / (name + ".pth")
                upstream_spec = {
                    "url": weight["upstream_url"],
                    "path": str(upstream.relative_to(output)),
                    "bytes": weight["upstream_bytes"],
                    "algorithm": "sha256",
                    "digest": weight["upstream_sha256"],
                }
                download(output, {}, upstream_spec)
                compact_dax(upstream, target, name, weight["upstream_sha256"])
                if not verified(target, weight):
                    target.unlink(missing_ok=True)
                    raise ValueError(f"Derived DAX checkpoint differs from catalog: {name}")
                upstream.unlink()
        statuses[name] = "verified" if verified(target, weight) else "missing_or_corrupt"
        print(name, statuses[name], flush=True)
    report = {"schema_version": 1, "research_only": True,
              "weights_in_release": False, "statuses": statuses,
              "all_ready": all(value == "verified" for value in statuses.values())}
    (output / "dxa-candidates-status.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["all_ready"] else 1)


if __name__ == "__main__":
    main()
