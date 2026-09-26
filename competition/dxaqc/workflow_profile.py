"""One checksum-pinned workflow contract shared by CLI and HTTP inference."""
import hashlib
import json
import os
from pathlib import Path

ARTIFACT_ENV = {'DXAQC_ANATOMY_MODEL', 'DXAQC_QUALITY_REVIEW_MODEL', 'DXAQC_PROJECTION_MODEL'}


def configure_profile(profile_path, baseline_path):
    profile_path = Path(profile_path)
    profile = json.loads(profile_path.read_text())
    if (profile.get('schema_version') != 1 or not isinstance(profile.get('profile_id'), str)
            or not profile['profile_id'].strip() or profile.get('clinical_validation') is not False):
        raise ValueError('Unsupported workflow profile')
    if hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest() != profile['baseline_bundle_sha256']:
        raise ValueError('Baseline model differs from pinned profile')
    code_root = Path(__file__).parent.parent.resolve()
    for name, digest in profile.get('code_sha256', {}).items():
        code = (code_root/name).resolve()
        if not code.is_relative_to(code_root) or hashlib.sha256(code.read_bytes()).hexdigest() != digest:
            raise ValueError('Inference code differs from pinned workflow')
    environment = {}
    for name, spec in profile['artifacts'].items():
        path = (profile_path.parent/spec['path']).resolve()
        if not path.is_relative_to(profile_path.parent.resolve()):
            raise ValueError('Profile artifact escapes its directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('Profile artifact checksum mismatch')
        environment[name] = str(path)
    if set(environment) != ARTIFACT_ENV:
        raise ValueError('Incomplete or unknown workflow components')
    # Mutate the environment only after every check passed.
    os.environ.update({**environment, 'DXAQC_WORKFLOW_PROFILE': str(profile_path.resolve())})
    return profile
