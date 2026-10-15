"""Assemble a model-only named Docker context for anatomical research inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_research.common import digest


def prepare(export, encoders, vertebrae, pelvis, landmarks, projection, output, baseline_profile, neck_template=None, visibility_cue=True,
            hip_display_landmarks=None, hip_display_template=None):
    if bool(hip_display_landmarks) != bool(hip_display_template):
        raise ValueError('Display landmark model and its template must be supplied together')
    required = [export / 'export.json', landmarks, projection]
    required.extend(source / filename for source in (vertebrae, pelvis)
                    for filename in ('dataset.json', 'plans.json', 'fold_0/checkpoint_best.pth'))
    required.extend(source for source in (neck_template, hip_display_landmarks, hip_display_template) if source is not None)
    for source in encoders:
        required.extend((source / 'config.json', source / 'preprocessor_config.json'))
    missing = [str(source) for source in required if not source.is_file()]
    if missing:
        raise FileNotFoundError('Missing model-only context inputs: ' + ', '.join(missing))
    if output.exists():
        raise ValueError('Choose a new asset directory')
    output.mkdir(parents=True)
    shutil.copytree(export, output / 'candidate')
    for name, source in zip(('dinov3-l', 'rad-dino'), encoders, strict=True):
        target = output / name
        target.mkdir()
        for path in source.iterdir():
            if path.suffix in ('.json', '.safetensors') or path.name in ('LICENSE', 'README.md'):
                shutil.copy2(path, target / path.name)
    for name, source in (('vertebrae', vertebrae), ('pelvis', pelvis)):
        (output / name / 'fold_0').mkdir(parents=True)
        for filename in ('dataset.json', 'plans.json', 'fold_0/checkpoint_best.pth'):
            shutil.copy2(source / filename, output / name / filename)
    for name, source in (('hip_landmarks.pt', landmarks), ('hip_projection.pt', projection)):
        shutil.copy2(source, output / 'candidate' / name)
    config = {'schema_version': 1, 'clinical_validation': False, 'visibility_cue': visibility_cue,
              'bank': 'bank', 'ranking_bank': 'ranking_bank',
              'decoders': 'decoders', 'feature_spec': 'feature_spec.json', 'baseline_profile': baseline_profile,
              'anatomy_model': '../vertebrae', 'pelvis_model': '../pelvis', 'encoders': ['../dinov3-l', '../rad-dino'],
              'pelvis_checkpoint_sha256': digest(pelvis / 'fold_0/checkpoint_best.pth'),
              'hip_landmarks': 'hip_landmarks.pt', 'hip_projection': 'hip_projection.pt',
              'hip_landmarks_sha256': digest(landmarks), 'hip_projection_sha256': digest(projection)}
    if neck_template is not None:
        shutil.copy2(neck_template, output / 'candidate/neck_template.json')
        config.update(neck_template='neck_template.json', neck_template_sha256=digest(neck_template))
    for name, source in (('hip_display_landmarks', hip_display_landmarks), ('hip_display_template', hip_display_template)):
        if source is not None:
            filename = name + source.suffix
            shutil.copy2(source, output / 'candidate' / filename)
            config[name], config[name + '_sha256'] = filename, digest(source)
    (output / 'candidate/runtime.json').write_text(json.dumps(config, indent=2) + '\n')
    inventory = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}
    (output / 'assets_manifest.json').write_text(json.dumps({'schema_version': 1, 'files': inventory,
        'includes_medical_images': False, 'includes_qc_targets': False, 'code_sha256': digest(Path(__file__)),
        'research_only': True, 'clinical_validation': False}, indent=2) + '\n')
    print(json.dumps({'files': len(inventory), 'output': str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export', 'vertebrae', 'pelvis', 'landmarks', 'projection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--encoders', type=Path, nargs=2, required=True)
    parser.add_argument('--baseline-profile', default='/app/models/workflow_1_12_0/profile.json')
    parser.add_argument('--neck-template', type=Path)
    parser.add_argument('--disable-spine-visibility', action='store_true')
    parser.add_argument('--hip-display-landmarks', type=Path)
    parser.add_argument('--hip-display-template', type=Path)
    args = parser.parse_args()
    prepare(args.export, args.encoders, args.vertebrae, args.pelvis, args.landmarks, args.projection,
            args.output, args.baseline_profile, args.neck_template, not args.disable_spine_visibility,
            args.hip_display_landmarks, args.hip_display_template)


if __name__ == '__main__':
    main()
