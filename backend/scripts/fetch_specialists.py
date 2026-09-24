"""Fetch pinned specialist sources/weights; inspect with --verify (no network).

Source snapshots and generic pretrained weights never activate a clinical model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from download_training_data import download, matches

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'docs/competition/specialist_sources.json'


def file_spec(artifact: dict) -> dict:
    return {**artifact, 'algorithm': 'sha256', 'digest': artifact['sha256']}


def extract_source(archive: Path, destination: Path, expected_sha256: str) -> dict:
    """Extract regular source files only, atomically; never follow archive links."""
    receipt = destination / '.osseo-source.json'
    if destination.exists():
        if receipt.is_file():
            existing = json.loads(receipt.read_text())
            if existing['archive_sha256'] == expected_sha256:
                return existing
        raise ValueError(f'Unmanaged source directory: {destination}')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Source archive checksum mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    skipped = []
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.extract-') as temp:
        staging = Path(temp) / 'source'
        staging.mkdir()
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            if len(members) > 50_000 or sum(m.size for m in members) > 2 * 1024**3:
                raise ValueError('Source archive exceeds extraction budget')
            roots, seen = set(), set()
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or not path.parts:
                    raise ValueError('Unsafe archive path')
                roots.add(path.parts[0])
                if len(roots) != 1:
                    raise ValueError('Expected one source archive root')
                parts = path.parts[1:]
                if not parts:
                    if not member.isdir():
                        raise ValueError('Source root must be a directory')
                    continue
                relative = PurePosixPath(*parts).as_posix()
                if relative in seen or relative == '.osseo-source.json':
                    raise ValueError('Duplicate or reserved archive path')
                seen.add(relative)
                if member.issym() or member.islnk():
                    skipped.append(relative)
                    continue
                if not (member.isfile() or member.isdir()):
                    raise ValueError('Unsupported archive entry')
                target = staging.joinpath(*parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.extractfile(member) as source, target.open('wb') as output:
                        shutil.copyfileobj(source, output)
        result = {'archive_sha256': expected_sha256, 'skipped_links': skipped}
        (staging / '.osseo-source.json').write_text(json.dumps(result, indent=2) + '\n')
        staging.rename(destination)
    return result


def prepare(catalog: dict, output: Path, *, verify: bool = False,
            only: set[str] | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    all_artifacts = catalog['artifacts']
    known_ids = {artifact['id'] for artifact in all_artifacts}
    if only is not None:
        unknown = only - known_ids
        if unknown:
            raise ValueError(f'Unknown artifact IDs: {", ".join(sorted(unknown))}')
        if not only:
            raise ValueError('Select at least one artifact')
        artifacts = [artifact for artifact in all_artifacts if artifact['id'] in only]
    else:
        artifacts = all_artifacts
    result = {'schema_version': 1, 'artifacts': []}
    if only is None:
        result['capabilities'] = catalog['capabilities']
    else:
        result['scope'] = {'selected_artifact_ids': [artifact['id'] for artifact in artifacts]}
    for artifact in artifacts:
        status = {'id': artifact['id'], 'path': artifact['path']}
        try:
            spec = file_spec(artifact)
            target = (output / spec['path']).resolve()
            if not target.is_relative_to(output.resolve()) or target == output.resolve():
                raise ValueError('Unsafe artifact destination')
            if verify:
                status['status'] = 'verified' if matches(target, spec) else 'missing_or_corrupt'
            else:
                status['status'] = download(output, {}, spec)
                if artifact['kind'] == 'source':
                    destination = output / 'sources' / artifact['id']
                    receipt = extract_source(target, destination, artifact['sha256'])
                    status['source_directory'] = str(destination.relative_to(output))
                    status['skipped_links'] = receipt['skipped_links']
        except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
            status.update(status='error', error=str(exc))
        result['artifacts'].append(status)
        print(f"{artifact['id']}: {status['status']}", flush=True)
    result['artifacts_ready'] = all(x['status'] in ('downloaded', 'verified') for x in result['artifacts'])
    result['clinical_models_ready'] = False
    # Only this tool owns this receipt; it never rewrites application configuration.
    report = output / ('assets-status.json' if only is None else 'selected-assets-status.json')
    with tempfile.NamedTemporaryFile('w', dir=output, delete=False) as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        temporary = Path(stream.name)
    temporary.replace(report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/specialists')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--only', nargs='+', metavar='ARTIFACT_ID',
                        help='Fetch/verify only these catalog artifacts and write a scoped receipt')
    args = parser.parse_args()
    report = prepare(json.loads(CATALOG.read_text()), args.output, verify=args.verify,
                     only=set(args.only) if args.only is not None else None)
    raise SystemExit(0 if report['artifacts_ready'] else 1)


if __name__ == '__main__':
    main()
