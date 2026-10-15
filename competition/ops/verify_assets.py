"""Verify the reviewed GPU model-only inventory before building or deploying."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(root, policy_path, code_root=None):
    root = root.resolve()
    policy = json.loads(policy_path.read_text())
    manifest_path = root/'assets_manifest.json'
    if sha256(manifest_path) != policy['assets_manifest_sha256']:
        raise ValueError('GPU assets are not the reviewed release; metric review and policy update are required')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('includes_medical_images') is not False or manifest.get('includes_qc_targets') is not False:
        raise ValueError('Expected a model-only artifact set')
    expected = set(manifest['files']) | {'assets_manifest.json'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError(f'Asset inventory differs: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    for name, digest in manifest['files'].items():
        path = root/name
        if path.is_symlink() or root not in path.resolve().parents or sha256(path) != digest:
            raise ValueError(f'Asset identity differs: {name}')
    code_root = (code_root or Path(__file__).resolve().parents[1]).resolve()
    for name, digest in policy['runtime_code_sha256'].items():
        path = code_root/name
        if code_root not in path.resolve().parents or sha256(path) != digest:
            raise ValueError(f'Inference code differs from reviewed release: {name}')
    return {'approved': True, 'verified_files': len(manifest['files']),
            'verified_runtime_modules': len(policy['runtime_code_sha256']),
            'assets_manifest_sha256': policy['assets_manifest_sha256'],
            'metric_scope': policy['metric_scope'], 'clinical_validation': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--policy', type=Path, default=Path(__file__).with_name('approved_gpu_assets.json'))
    p.add_argument('--code-root', type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.root, a.policy, a.code_root)))


if __name__ == '__main__':
    main()
