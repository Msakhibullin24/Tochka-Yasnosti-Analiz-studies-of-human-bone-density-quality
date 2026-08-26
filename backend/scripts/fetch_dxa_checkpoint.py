from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path


FILE_ID = "1KSF1LceRbd3YttfgGk70JPoLobHW2zkr"
URL = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"
EXPECTED_SIZE = 277_761_941
EXPECTED_SHA256 = "92a90627ecde530a2c259fa864c578806cdda5b0718a26180e891779ad453ced"
TARGET = Path(__file__).resolve().parents[1] / "models" / "checkpoint.pth"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size == EXPECTED_SIZE:
        digest = sha256(TARGET)
        if digest == EXPECTED_SHA256:
            print(f"Checkpoint already exists and is verified: {TARGET}")
            return 0
        print(f"Existing checkpoint checksum mismatch ({digest}); downloading a verified copy.")
    with tempfile.NamedTemporaryFile(prefix=".checkpoint-", dir=TARGET.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        print(f"Downloading upstream checkpoint to {TARGET} ({EXPECTED_SIZE / 1024 / 1024:.1f} MiB)...")
        with urllib.request.urlopen(URL) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        size = temporary_path.stat().st_size
        if size != EXPECTED_SIZE:
            raise RuntimeError(f"Unexpected checkpoint size: {size}; expected {EXPECTED_SIZE}")
        digest = sha256(temporary_path)
        if digest != EXPECTED_SHA256:
            raise RuntimeError(f"Unexpected checkpoint sha256: {digest}; expected {EXPECTED_SHA256}")
        os.replace(temporary_path, TARGET)
        print(f"Saved {TARGET}")
        print(f"sha256={digest}")
        print("The checkpoint is distributed outside the GitHub repository. Confirm its licensing before redistribution.")
        return 0
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
