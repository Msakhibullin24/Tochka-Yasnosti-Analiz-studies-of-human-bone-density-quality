"""Immutable local input snapshots for repeatable analysis, including companion DICOM."""
from pathlib import Path
import hashlib
import os
import shutil
import zipfile


def retain_input(source: Path, job_dir: Path) -> Path:
    target = job_dir / 'source.zip'
    if target.exists():
        raise FileExistsError('input snapshot already exists')
    temporary = job_dir / '.source.zip.tmp'
    try:
        if source.is_file() and source.suffix.lower() == '.zip':
            shutil.copyfile(source, temporary)
        else:
            files = sorted(source.rglob('*')) if source.is_dir() else [source]
            with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in files:
                    if path.is_symlink():
                        raise ValueError('input snapshots do not follow symbolic links')
                    if path.is_file():
                        archive.write(path, path.relative_to(source) if source.is_dir() else path.name)
            if not source.exists():
                raise FileNotFoundError(source)
        os.replace(temporary, target)
        with target.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        (job_dir / 'source.sha256').write_text(digest + '\n')
        return target
    finally:
        temporary.unlink(missing_ok=True)
