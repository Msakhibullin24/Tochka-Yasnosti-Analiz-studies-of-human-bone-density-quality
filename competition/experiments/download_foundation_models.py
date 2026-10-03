"""Download the four approved research checkpoints sequentially from their owners.

Use Hugging Face login or HF_TOKEN in the environment. A gated repository still
requires the account to accept its terms on the model page. Tokens are never
written to the manifest. Run again to resume an interrupted download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError

REPOS = (
    "google/medsiglip-448",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "facebook/sam3",
)
PATTERNS = ("*.json", "*.safetensors", "*.model", "README.md")


def sha256(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/specialists/models"))
    parser.add_argument("--repo", action="append", choices=REPOS,
                        help="Download selected repository; omit to process all four")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    results = {}
    failed = False
    for repo in args.repo or REPOS:
        target = root / repo.split("/")[1]
        print(f"Checking {repo}", flush=True)
        try:
            revision = api.model_info(repo).sha
            snapshot_download(repo_id=repo, revision=revision, local_dir=target,
                              allow_patterns=PATTERNS, max_workers=2)
            weights = sorted(target.glob("*.safetensors"))
            if not weights:
                raise ValueError("No safetensors checkpoint was downloaded")
            if (target / "model.safetensors.index.json").exists():
                index = json.loads((target / "model.safetensors.index.json").read_text())
                if {p.name for p in weights} != set(index["weight_map"].values()):
                    raise ValueError("Sharded checkpoint is incomplete")
            files = []
            for path in weights:
                digest = sha256(path)
                meta = target / ".cache/huggingface/download" / (path.name + ".metadata")
                if meta.exists():
                    etag = meta.read_text().splitlines()[1].strip('"')
                    if len(etag) == 64 and digest != etag:
                        raise ValueError(f"SHA256 mismatch: {path.name}")
                files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})
            results[repo] = {"status": "downloaded", "revision": revision,
                             "directory": str(target), "weights": files}
            print(f"Verified {repo}: {sum(x['bytes'] for x in files):,} weight bytes", flush=True)
        except GatedRepoError:
            failed = True
            results[repo] = {"status": "access_denied", "model_page": "https://huggingface.co/" + repo}
            print(f"Account lacks gated access: {repo}", flush=True)
        except Exception as exc:
            failed = True
            results[repo] = {"status": "error", "type": type(exc).__name__}
            print(f"Download failed: {repo}: {type(exc).__name__}", flush=True)
        (root / "download_manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
