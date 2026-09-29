"""Package the complete offline workflow, including nested anatomy weights.

Deriving a new profile records changed code explicitly. It does not relabel
previous model evaluation as independent validation of the new inference path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from dxaqc.pipeline import MODEL_PATH
from dxaqc.workflow_profile import ARTIFACT_ENV, configure_profile

ROOT = Path(__file__).resolve().parent
NAMES = {'DXAQC_ANATOMY_MODEL': 'anatomy.json',
         'DXAQC_QUALITY_REVIEW_MODEL': 'quality_review.joblib',
         'DXAQC_PROJECTION_MODEL': 'projection.joblib'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_source(directory, spec):
    path = (directory/spec['path']).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('Source artifact escapes profile directory')
    if digest(path) != spec['sha256']:
        raise ValueError('Source artifact checksum mismatch')
    return path


def pack(source, output, profile_id):
    if output.exists():
        raise ValueError('Choose a new workflow output directory')
    original = json.loads(source.read_text())
    if (original.get('schema_version') != 1 or original.get('clinical_validation') is not False
            or set(original['artifacts']) != ARTIFACT_ENV
            or original['baseline_bundle_sha256'] != digest(MODEL_PATH)):
        raise ValueError('Incompatible source workflow')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.workflow-') as temporary:
        directory = Path(temporary)
        artifacts = {}
        for name, spec in original['artifacts'].items():
            path = checked_source(source.parent, spec)
            target = directory/NAMES[name]
            if name == 'DXAQC_ANATOMY_MODEL':
                anatomy = json.loads(path.read_text())
                if anatomy.get('schema_version') != 1 or set(anatomy['models']) != {'spine', 'hip'}:
                    raise ValueError('Complete spine/hip anatomy portfolio required')
                for group, member in anatomy['models'].items():
                    checkpoint = checked_source(path.parent, member)
                    shutil.copy2(checkpoint, directory/(group+'.pt'))
                    member['path'] = group+'.pt'
                target.write_text(json.dumps(anatomy, indent=2)+'\n')
            else:
                shutil.copy2(path, target)
            artifacts[name] = {'path': target.name, 'sha256': digest(target)}
        files = sorted((ROOT/'dxaqc').glob('*.py'))
        files += [ROOT/'run_no_new_labels.py', ROOT/'pack_workflow.py']
        profile = {'schema_version': 1, 'profile_id': profile_id,
                   'baseline_bundle_sha256': digest(MODEL_PATH),
                   'clinical_validation': False, 'status': 'engineering_delivery',
                   'source_profile_sha256': digest(source),
                   'source_profile_id': original['profile_id'],
                   'artifacts': artifacts,
                   'code_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
                   'limitations': ['Anatomical candidates are not verified clinical landmarks.',
                                   'Secondary type review changes sensitivity and probability scores.',
                                   'Historical OOF is not independent evaluation of this delivery.']}
        manifest = directory/'profile.json'
        manifest.write_text(json.dumps(profile, indent=2)+'\n')
        # Verify the nested weight closure with the actual runtime loaders.
        from dxaqc.learned_anatomy import AnatomyPortfolio
        AnatomyPortfolio(directory/'anatomy.json')
        previous = {name: os.environ.get(name) for name in ARTIFACT_ENV | {'DXAQC_WORKFLOW_PROFILE'}}
        try:
            configure_profile(manifest, MODEL_PATH)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        # mkdtemp is private (0700). Shipped nonsecret model assets must be
        # readable by the container's unprivileged inference user.
        for artifact in directory.iterdir():
            artifact.chmod(0o644)
        directory.chmod(0o755)
        directory.rename(output)
    return profile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-id', required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.source, args.output, args.profile_id), indent=2))
