from __future__ import annotations

import shutil
import subprocess
import tempfile
import re
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Iterator

from .parser import ApexFormatError


MAX_ARCHIVE_ENTRIES = 100_000
MAX_UNPACKED_BYTES = 5 * 1024 * 1024 * 1024
APEX_SCAN_NAME = re.compile(r"\.[pr][a-z0-9]{2}$", re.IGNORECASE)


def _archive_entries(archive: Path, executable: str) -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [executable, "l", "-ba", "-slt", str(archive)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ApexFormatError(f"Unable to inspect archive {archive.name}: {error}") from error
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in result.stdout.splitlines() + [""]:
        if not line.strip():
            if current:
                entries.append(current)
                current = {}
            continue
        key, separator, value = line.partition(" = ")
        if separator:
            current[key] = value
    return entries


def _validate_archive(archive: Path, executable: str) -> None:
    entries = _archive_entries(archive, executable)
    if not entries:
        raise ApexFormatError(f"Archive {archive.name} contains no entries")
    if len(entries) > MAX_ARCHIVE_ENTRIES:
        raise ApexFormatError(f"Archive {archive.name} exceeds the entry limit")
    total_size = 0
    for entry in entries:
        raw_name = entry.get("Path", "").replace("\\", "/")
        name = PurePosixPath(raw_name)
        drive_like = bool(name.parts and name.parts[0].endswith(":"))
        if not raw_name or name.is_absolute() or drive_like or ".." in name.parts:
            raise ApexFormatError(f"Archive {archive.name} contains unsafe path {raw_name!r}")
        if entry.get("Encrypted") == "+":
            raise ApexFormatError(f"Archive {archive.name} contains an encrypted entry: {raw_name}")
        if entry.get("Symbolic Link") or entry.get("Hard Link"):
            raise ApexFormatError(f"Archive {archive.name} contains a link entry: {raw_name}")
        try:
            total_size += int(entry.get("Size", "0") or 0)
        except ValueError as error:
            raise ApexFormatError(f"Archive {archive.name} has invalid size metadata") from error
    if total_size > MAX_UNPACKED_BYTES:
        raise ApexFormatError(f"Archive {archive.name} exceeds the unpacked-size limit")


@contextmanager
def materialize_source(source: Path) -> Iterator[Path]:
    source = source.expanduser().resolve()
    if source.is_dir():
        if any(
            path.suffix.lower() != ".rar" and APEX_SCAN_NAME.search(path.name)
            for path in source.rglob("*")
            if path.is_file()
        ):
            yield source
            return
        archives = sorted(path for path in source.rglob("*.rar") if path.is_file())
        if len(archives) == 1:
            with materialize_source(archives[0]) as extracted:
                yield extracted
            return
        if len(archives) > 1:
            raise ApexFormatError(
                f"Directory contains {len(archives)} RAR archives; select one archive explicitly"
            )
        yield source
        return
    if not source.is_file():
        raise ApexFormatError(f"Input does not exist: {source}")
    if source.suffix.lower() != ".rar":
        raise ApexFormatError("Input must be an extracted directory or a .rar archive")
    executable = shutil.which("7z")
    if not executable:
        raise ApexFormatError("7z is required to inspect and extract RAR archives")
    _validate_archive(source, executable)
    with tempfile.TemporaryDirectory(prefix="osseo-apex-") as temporary:
        target = Path(temporary).resolve()
        try:
            subprocess.run(
                [executable, "x", "-y", "-bd", "-bso0", "-bsp0", str(source), f"-o{target}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ApexFormatError(f"Unable to extract archive {source.name}: {error}") from error
        for path in target.rglob("*"):
            if path.is_symlink():
                raise ApexFormatError(f"Archive extraction produced a symbolic link: {path.name}")
            try:
                path.resolve().relative_to(target)
            except ValueError as error:
                raise ApexFormatError(f"Archive extraction escaped its temporary directory: {path}") from error
        yield target
