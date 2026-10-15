"""Stream-verify compressed Docker archive CRC, blob hashes and expected tags."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-tags', nargs='+', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sha = hashlib.sha256()
    with args.archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    if sha.hexdigest() != args.expected_sha256:
        raise ValueError('Compressed archive checksum differs')
    manifest, blobs, members = None, 0, 0
    with gzip.open(args.archive, 'rb') as decompressed:
        with tarfile.open(fileobj=decompressed, mode='r|') as archive:
            for member in archive:
                members += 1
                if not member.isfile():
                    continue
                stream = archive.extractfile(member)
                if member.name == 'manifest.json':
                    manifest = json.load(stream)
                    continue
                match = re.fullmatch(r'blobs/sha256/([0-9a-f]{64})', member.name)
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
                if match:
                    if digest.hexdigest() != match[1]:
                        raise ValueError('Invalid Docker blob: ' + member.name)
                    blobs += 1
        # Read through the gzip trailer even if tar ended at an earlier zero block.
        while decompressed.read(8 * 1024 * 1024):
            pass
    if manifest is None:
        raise ValueError('Docker manifest is missing')
    tags = sorted({tag for entry in manifest for tag in entry.get('RepoTags', [])})
    if not set(args.expected_tags) <= set(tags):
        raise ValueError('Expected image tags are missing')
    report = {'archive': str(args.archive), 'bytes': args.archive.stat().st_size,
              'sha256': sha.hexdigest(), 'gzip_crc_valid': True, 'tar_members': members,
              'verified_sha256_blobs': blobs, 'images': len(manifest), 'tags': tags,
              'docker_load_performed': False, 'verification': 'complete compressed checksum, gzip CRC and content-addressed blobs'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
