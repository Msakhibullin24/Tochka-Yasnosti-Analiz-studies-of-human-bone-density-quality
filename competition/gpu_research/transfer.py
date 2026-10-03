"""Create/verify a private data ZIP; source data and review pixels stay outside Git."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def stream_sha(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def build(dataset, output, review=None, test_dataset=None, task_documents=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new archive path')
    roots = {'dataset': Path(dataset).resolve()}
    if review:
        roots['review'] = Path(review).resolve()
    if test_dataset:
        roots['test-dataset'] = Path(test_dataset).resolve()
    if task_documents:
        roots['task-documents'] = Path(task_documents).resolve()
    if any(not p.is_dir() or output.is_relative_to(p) for p in roots.values()):
        raise ValueError('Source folders must exist and archive must be outside them')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    if temporary.exists():
        raise ValueError('An unfinished archive exists; choose a new output path')
    manifest = {'schema_version': 1, 'private_local_data': True, 'files': {}}
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, root in roots.items():
                for path in sorted(root.rglob('*')):
                    if path.is_symlink():
                        raise ValueError('Symlinks are not allowed in the data bundle')
                    if not path.is_file():
                        continue
                    if name == 'task-documents' and path.suffix.lower() not in {'.pdf', '.docx', '.xlsx', '.txt', '.md'}:
                        continue
                    relative = f'{name}/{path.relative_to(root).as_posix()}'
                    with path.open('rb') as source, archive.open(relative, 'w') as target:
                        value, size = hashlib.sha256(), 0
                        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                            value.update(block)
                            size += len(block)
                            target.write(block)
                    manifest['files'][relative] = {'sha256': value.hexdigest(), 'bytes': size}
            if not manifest['files']:
                raise ValueError('Empty bundle')
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        verify(temporary)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with output.open('rb') as stream:
        checksum = stream_sha(stream)
    output.with_suffix(output.suffix + '.sha256').write_text(f'{checksum}  {output.name}\n')
    return {'archive': str(output), 'sha256': checksum, 'files': len(manifest['files'])}


def verify(path, extract=None, max_bytes=20 * 1024**3):
    if extract and Path(extract).exists():
        raise ValueError('Extraction destination must be new')
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        names = [item.filename for item in items]
        if len(set(names)) != len(names) or len(names) > 100000 or sum(i.file_size for i in items) > max_bytes:
            raise ValueError('Duplicate ZIP paths or archive size limit exceeded')
        for item in items:
            path_parts = PurePosixPath(item.filename)
            if (path_parts.is_absolute() or '..' in path_parts.parts or '\\' in item.filename
                    or ':' in item.filename or stat.S_ISLNK(item.external_attr >> 16)
                    or item.is_dir() or str(path_parts) != item.filename):
                raise ValueError('Unsafe archive member')
        manifest_info = archive.getinfo('manifest.json')
        if manifest_info.file_size > 32 * 1024**2:
            raise ValueError('Manifest size limit exceeded')
        manifest = json.loads(archive.read('manifest.json'))
        files = manifest.get('files', {})
        if manifest.get('schema_version') != 1 or set(names) != set(files) | {'manifest.json'}:
            raise ValueError('Archive inventory differs from manifest')
        for name, expected in files.items():
            with archive.open(name) as stream:
                checksum = stream_sha(stream)
            if checksum != expected['sha256'] or archive.getinfo(name).file_size != expected['bytes']:
                raise ValueError('Archive member checksum/size differs')
        # No files written until every path and checksum has been checked.
        if extract:
            destination = Path(extract).resolve()
            destination.mkdir(parents=True)
            for item in items:
                target = destination / item.filename
                if not target.resolve().is_relative_to(destination):
                    raise ValueError('Archive member escapes destination')
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open('xb') as result:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        result.write(block)
    return {'verified': True, 'files': len(files), 'extract': str(extract) if extract else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    pack = sub.add_parser('build')
    pack.add_argument('--dataset', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    pack.add_argument('--review', type=Path)
    pack.add_argument('--test-dataset', type=Path)
    pack.add_argument('--task-documents', type=Path, help='Include only PDF/DOCX/XLSX/TXT/MD from task folder')
    check = sub.add_parser('verify')
    check.add_argument('--archive', type=Path, required=True)
    check.add_argument('--extract', type=Path)
    a = p.parse_args()
    report = build(a.dataset, a.output, a.review, a.test_dataset, a.task_documents) if a.command == 'build' else verify(a.archive, a.extract)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
