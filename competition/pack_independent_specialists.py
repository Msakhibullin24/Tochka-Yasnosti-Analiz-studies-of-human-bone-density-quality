"""Package three independently trained QC models without sharing encoders or outputs."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from dxaqc.specialist_qc import IndependentSpecialists, SpecialistQC, digest

ROLES = {
    'spine_convnext': ('spine', 'lumbar_spine_quality_and_criteria'),
    'hip_convnext': ('hip', 'proximal_femur_position_and_roi'),
    'general_efficientnet': ('all', 'general_dxa_quality_crosscheck'),
}


def pack(spine: Path, hip: Path, general: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Choose a new portfolio directory')
    sources = {'spine_convnext': spine, 'hip_convnext': hip, 'general_efficientnet': general}
    models = {name: SpecialistQC(path) for name, path in sources.items()}
    for name, model in models.items():
        scope, _ = ROLES[name]
        if model.metadata.get('region_scope', 'all') != scope:
            raise ValueError(f'{name} has the wrong anatomical scope')
    label_hashes = {model.metadata.get('labels_sha256') for model in models.values()}
    label_hash = next(iter(label_hashes))
    if (len(label_hashes) != 1 or not isinstance(label_hash, str)
            or len(label_hash) != 64 or any(c not in '0123456789abcdef' for c in label_hash.lower())):
        raise ValueError('Independent models must use the same label revision for comparison')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.portfolio-') as temporary:
        root = Path(temporary)
        spec = {'schema_version': 1, 'kind': 'independent', 'mode': 'shadow',
                'affects_decision': False, 'members': {}}
        for name, source in sources.items():
            target = root / name
            target.mkdir()
            for filename in ('model.json', 'model.ts'):
                shutil.copy2(source / filename, target / filename)
            scope, responsibility = ROLES[name]
            spec['members'][name] = {'path': name, 'metadata_sha256': digest(target / 'model.json'),
                                     'region_scope': scope, 'responsibility': responsibility}
        (root / 'portfolio.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
        IndependentSpecialists(root)
        root.rename(output)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spine', 'hip', 'general', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    spec = pack(args.spine, args.hip, args.general, args.output)
    print(json.dumps({'output': str(args.output), 'members': list(spec['members'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
