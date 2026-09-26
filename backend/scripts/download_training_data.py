"""Download selected public datasets with pinned checksums; Python stdlib only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "docs/competition/download_sources.json"


def matches(path: Path, spec: dict) -> bool:
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        return False
    algorithm = spec["algorithm"]
    digest = hashlib.new("sha1" if algorithm == "git-sha1" else algorithm)
    if algorithm == "git-sha1":
        digest.update(f"blob {spec['bytes']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == spec["digest"]


def download(directory: Path, source: dict, spec: dict) -> str:
    directory = directory.resolve()
    target = (directory / spec["path"]).resolve()
    if not target.is_relative_to(directory) or target == directory:
        raise ValueError(f"Unsafe destination: {spec['path']}")
    if target.exists():
        if matches(target, spec):
            return "verified"
        raise ValueError(f"Existing file differs from source; inspect or move it: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = spec.get("url") or source["base_url"] + quote(spec["path"])
    if not url.startswith("https://"):
        raise ValueError("Downloads must use HTTPS")
    temporary = target.with_name(target.name + ".part")
    if temporary.is_symlink():
        raise ValueError(f"Unsafe partial destination: {temporary}")
    for attempt in range(3):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset > spec["bytes"]:
                raise ValueError(f"Partial exceeds expected size: {spec['path']}")
            if offset == spec["bytes"]:
                if not matches(temporary, spec):
                    raise ValueError(f"Size/checksum mismatch: {spec['path']}")
                temporary.replace(target)
                return "downloaded"
            headers = {"User-Agent": "OsseoAI-data-download/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                resumed = offset and getattr(response, "status", None) == 206
                if resumed:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{spec['bytes']}"):
                        raise ValueError("Invalid Content-Range in resumed download")
                # A server may ignore Range. In that case restart, never append
                # a complete response to a partial file.
                with temporary.open("ab" if resumed else "wb") as output:
                    total = offset if resumed else 0
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        total += len(chunk)
                        if total > spec["bytes"]:
                            raise ValueError(f"Response exceeds expected size: {spec['path']}")
                        output.write(chunk)
            if total != spec["bytes"]:
                raise OSError(f"Interrupted response: {total}/{spec['bytes']} bytes for {spec['path']}")
            if not matches(temporary, spec):
                raise ValueError(f"Size/checksum mismatch: {spec['path']}")
            temporary.replace(target)
            return "downloaded"
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError):
                temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("Unreachable")


def main() -> None:
    catalog = json.loads(CATALOG.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["starter", *catalog], default="starter")
    parser.add_argument("--output", type=Path, default=ROOT / "data/external")
    parser.add_argument("--list", action="store_true", help="Print sizes and licenses without downloading")
    args = parser.parse_args()
    names = list(catalog) if args.list else (["ramathibodi", "hipray"] if args.dataset == "starter" else [args.dataset])
    for name in names:
        source = catalog[name]
        size = sum(file["bytes"] for file in source["files"])
        print(f"{name}: {size / 1_000_000:.2f} MB; {source['license']}; {source['description']}", flush=True)
        if args.list:
            continue
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        needed = sum(file["bytes"] for file in source["files"] if not (directory / file["path"]).exists())
        if shutil.disk_usage(directory).free < needed + 64 * 1024 * 1024:
            raise OSError(f"Insufficient disk space: {directory}")
        # Four workers suffice for this small file collection; no extra download dependency.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for count, _ in enumerate(pool.map(lambda file: download(directory, source, file), source["files"]), 1):
                if count % 20 == 0 or count == len(source["files"]):
                    print(f"  checked {count}/{len(source['files'])}", flush=True)
        print(f"Ready: {directory.resolve()}", flush=True)


if __name__ == "__main__":
    main()
