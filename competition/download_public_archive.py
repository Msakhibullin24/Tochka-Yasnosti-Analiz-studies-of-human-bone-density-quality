"""Resume a public archive with bounded HTTP ranges, then verify ZIP CRC/SHA-256.

Only a contiguous downloaded prefix is kept in .part. Parallel range fragments
are temporary and never masquerade as a complete download. Sources without a
publisher checksum retain an explicit local-integrity provenance record.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import Request, urlopen

from audit_downloaded_datasets import archive_inventory, fingerprint


def fetch_range(url, start, end, size, destination):
    """Reject ignored ranges and mismatched bodies before extending the prefix."""
    for attempt in range(3):
        try:
            request = Request(url, headers={'User-Agent': 'OsseoAI-data-download/1',
                                           'Range': f'bytes={start}-{end}',
                                           'Accept-Encoding': 'identity'})
            with urlopen(request, timeout=60) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                    raise ValueError('Server did not honour the exact HTTP range')
                count = 0
                with destination.open('wb') as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        count += len(block)
                        if count > end - start + 1:
                            raise ValueError('HTTP range body exceeds expected size')
                        stream.write(block)
                if count != end - start + 1:
                    raise OSError('Interrupted HTTP range body')
            return destination
        except (OSError, ValueError):
            destination.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def download(url, target, size, workers=4, chunk_size=8 * 1024 * 1024):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name + '.download.lock')
    if lock.is_symlink():
        raise ValueError('Refusing a symlink lock')
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another downloader owns this archive') from exc
        return _download(url, target, size, workers, chunk_size)


def _download(url, target, size, workers, chunk_size):
    if not url.startswith('https://') or size <= 0 or not 1 <= workers <= 8 or chunk_size <= 0:
        raise ValueError('Invalid download parameters')
    target = Path(target)
    partial = target.with_name(target.name + '.part')
    if target.is_symlink() or partial.is_symlink():
        raise ValueError('Refusing a symlink destination')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != size:
            raise ValueError('Existing archive size differs; original preserved')
        details = archive_inventory(target)
        return {'status': 'verified', 'sha256': fingerprint(target), 'archive': details}
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError('Partial archive exceeds expected size')
    needed = size - offset + workers * chunk_size
    if shutil.disk_usage(target.parent).free < needed + 15 * 1024**3:
        raise OSError('Keep at least 15 GiB free after download and temporary fragments')
    # Submit one window at a time: at most workers * chunk_size scratch bytes.
    with tempfile.TemporaryDirectory(prefix='.ranges-', dir=target.parent) as temporary:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while offset < size:
                window = []
                for index in range(workers):
                    start = offset + index * chunk_size
                    if start >= size:
                        break
                    end = min(size - 1, start + chunk_size - 1)
                    destination = Path(temporary) / str(index)
                    window.append((start, end, pool.submit(fetch_range, url, start, end, size, destination)))
                with partial.open('ab') as output:
                    for start, end, future in window:
                        fragment = future.result()
                        if partial.stat().st_size != start:
                            raise ValueError('Partial changed during download; another writer may be active')
                        with fragment.open('rb') as stream:
                            shutil.copyfileobj(stream, output)
                        output.flush()
                        fragment.unlink()
                        offset = end + 1
                print(f'{target.name}: {offset}/{size} bytes', flush=True)
    if partial.stat().st_size != size:
        raise ValueError('Incomplete archive')
    details = archive_inventory(partial)
    sha256 = fingerprint(partial)
    partial.rename(target)
    return {'status': 'downloaded', 'sha256': sha256, 'archive': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--file', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/external')
    args = parser.parse_args()
    catalog_path = Path(__file__).resolve().parents[1] / 'docs/competition/additional_dataset_acquisition.json'
    manifest = json.loads(catalog_path.read_text())
    spec = next(item for item in manifest['files'] if item['dataset'] == args.dataset and item['path'] == args.file)
    target = (args.output / spec['dataset'] / spec['path']).resolve()
    if not target.is_relative_to(args.output.resolve()):
        raise ValueError('Unsafe manifest path')
    details = download(spec['url'], target, spec['bytes'], args.workers)
    receipt = {**spec, **details, 'digest_origin': 'Local SHA-256 and ZIP CRC; not a publisher signature'}
    target.with_name(target.name + '.provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'file': str(target), 'status': details['status'], 'sha256': details['sha256']}), flush=True)


if __name__ == '__main__':
    main()
