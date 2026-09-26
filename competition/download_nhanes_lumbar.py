"""Download a reproducible NHANES II lateral lumbar subset from the CDC index.

CDC publishes sizes but no checksums here. Local digests record integrity only.
Images have source projection labels, no numbered vertebral or QC ground truth.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess


SOURCE = 'https://ftp.cdc.gov/pub/NHANES/XRays/Nhanes2/'


def inventory(html):
    rows = re.findall(r'(\d+)\s+<A HREF="(/pub/NHANES/XRays/Nhanes2/L\d+\.tiff)"', html, re.I)
    files = [{'path': Path(url).name, 'bytes': int(size), 'url': 'https://ftp.cdc.gov'+url}
             for size, url in rows if int(size) >= 8]
    if not files or len({f['path'] for f in files}) != len(files):
        raise ValueError('Invalid or empty CDC lumbar inventory')
    return files


def fingerprint(path, size):
    if path.stat().st_size != size:
        raise ValueError('CDC source size differs')
    with path.open('rb') as stream:
        if stream.read(4) not in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'):
            raise ValueError('Response is not a TIFF image')
        stream.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(root, spec):
    path = root/spec['path']
    partial = path.with_suffix('.tiff.part')
    if path.is_symlink() or partial.is_symlink():
        raise ValueError('Unsafe destination symlink')
    if path.exists():
        return {**spec, 'status': 'downloaded', 'sha256': fingerprint(path, spec['bytes'])}
    if partial.exists() and partial.stat().st_size == spec['bytes']:
        digest = fingerprint(partial, spec['bytes'])
        partial.rename(path)
        return {**spec, 'status': 'downloaded', 'sha256': digest}
    # curl validates HTTP status, resumes partial responses and retries network errors.
    result = subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                             '--connect-timeout', '15', '--max-time', '1200', '--retry', '3',
                             '--speed-time', '90', '--speed-limit', '1024',
                             '--continue-at', '-', spec['url'], '--output', str(partial)],
                            capture_output=True, text=True)
    if result.returncode:
        return {**spec, 'status': 'download_error', 'error': result.stderr[-1000:]}
    try:
        digest = fingerprint(partial, spec['bytes'])
    except (OSError, ValueError) as exc:
        return {**spec, 'status': 'invalid', 'error': str(exc)}
    partial.rename(path)
    return {**spec, 'status': 'downloaded', 'sha256': digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--count', type=int, default=1000)
    args = parser.parse_args()
    files = inventory(args.index.read_text())
    if not 1 <= args.count <= len(files):
        parser.error('count must be between 1 and the source cohort size')
    selected = sorted(random.Random(17).sample(files, args.count), key=lambda f: f['path'])
    args.output.mkdir(parents=True, exist_ok=True)
    needed = sum(f['bytes'] for f in selected if not (args.output/f['path']).exists())
    if shutil.disk_usage(args.output).free < needed + 15*1024**3:
        raise OSError('Keep at least 15 GiB free after downloads')
    report_path = args.output/'acquisition.json'
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    previous_hashes = {r['path']: r['sha256'] for r in previous.get('files', []) if 'sha256' in r}
    report = {'source': SOURCE, 'source_cohort_usable_images': len(files),
              'excluded_source_entries': [url for size, url in re.findall(
                  r'(\d+)\s+<A HREF="(/pub/NHANES/XRays/Nhanes2/L\d+\.tiff)"', args.index.read_text(), re.I) if int(size) < 8],
              'source_cohort_bytes': sum(f['bytes'] for f in files),
              'selection': f'{args.count} uniformly sampled source files; seed 17; disk-bounded subset, not full cohort',
              'reference_basis': 'CDC L-prefix: lateral lumbar; radiological reading files not released',
              'clinical_validation': False, 'digest_origin': 'Local SHA-256; no publisher checksum',
              'index_sha256': hashlib.sha256(args.index.read_bytes()).hexdigest(), 'files': []}
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(download, args.output, spec) for spec in selected]
        for future in as_completed(pending):
            row = future.result()
            if row['path'] in previous_hashes and row.get('sha256') != previous_hashes[row['path']]:
                row = {**row, 'status': 'changed_since_previous_download'}
            report['files'].append(row)
            report['files'].sort(key=lambda item: item['path'])
            temporary = report_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(report, indent=2)+'\n')
            temporary.replace(report_path)
            if len(report['files']) % 20 == 0:
                print(f"Recorded {len(report['files'])}/{len(selected)} files", flush=True)
    failures = sum(r['status'] != 'downloaded' for r in report['files'])
    print(f"Finished: {len(selected)-failures} downloaded, {failures} failed", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
