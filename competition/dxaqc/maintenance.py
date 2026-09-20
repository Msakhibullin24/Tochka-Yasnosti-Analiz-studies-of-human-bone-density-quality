"""Offline backup/restore. Stop the service before either operation.

python -m dxaqc.maintenance backup --root /work/dxaqc-jobs --archive backup.zip
python -m dxaqc.maintenance restore --root /new/work/dxaqc-jobs --archive backup.zip
"""
from __future__ import annotations

import argparse
import fcntl
import os
import sqlite3
import tempfile
import zipfile
from pathlib import Path


def backup(root: Path, archive: Path):
    root = root.resolve()
    archive = archive.resolve()
    if not (root / 'jobs.sqlite3').is_file():
        raise ValueError('job database not found')
    if archive == root or root in archive.parents:
        raise ValueError('backup must be outside the data directory')
    with open(root / 'worker.lock', 'a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=archive.parent, prefix='.backup-')
        os.close(fd)
        try:
            with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
                for p in sorted(root.rglob('*')):
                    if p.is_file() and not p.is_symlink() and p.name != 'worker.lock':
                        z.write(p, p.relative_to(root))
            os.replace(tmp, archive)
        finally:
            Path(tmp).unlink(missing_ok=True)


def restore(root: Path, archive: Path):
    if root.exists() and any(root.iterdir()):
        raise ValueError('restore destination must be empty')
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent, prefix='.restore-') as directory:
        staging = Path(directory)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                p = (staging / member.filename).resolve()
                if staging.resolve() not in p.parents or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('unsafe backup member')
            z.extractall(staging)
        if not (staging / 'jobs.sqlite3').is_file():
            raise ValueError('backup has no database')
        db = sqlite3.connect(staging / 'jobs.sqlite3')
        try:
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('backup database is corrupt')
            # Job paths are relocatable when their artifacts are inside the backed-up root.
            for job_id, out, source, upload in db.execute('SELECT id,out,input,upload_dir FROM jobs').fetchall():
                old_root = Path(out).parent.parent
                values = []
                for value in (out, source, upload):
                    if value is None:
                        values.append(None)
                    else:
                        try:
                            values.append(str(root.resolve() / Path(value).relative_to(old_root)))
                        except ValueError:
                            values.append(value)  # externally mounted input, not part of backup
                db.execute('UPDATE jobs SET out=?,input=?,upload_dir=? WHERE id=?', (*values, job_id))
            db.commit()
        finally:
            db.close()
        if root.exists():
            root.rmdir()
        os.replace(staging, root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['backup', 'restore'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--archive', required=True, type=Path)
    args = p.parse_args()
    try:
        (backup if args.action == 'backup' else restore)(args.root, args.archive)
    except BlockingIOError:
        p.error('stop the service before backing up its data')


if __name__ == '__main__':
    main()
