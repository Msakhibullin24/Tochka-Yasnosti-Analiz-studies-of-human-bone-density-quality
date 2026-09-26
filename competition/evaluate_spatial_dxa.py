"""Frozen BUU L1-L4 predictions against named public DXA bone-area ROI centers.

ROIs are weaker references than anatomical contours. No assignment or renumbering
using the targets is permitted. All four numbered predictions count in the metric.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dxaqc.dicom_io import read_dxa
from evaluate_external_roi_masks import read_seg_nrrd
from evaluate_organizer_dataset import sha256
from predict_spatial_anatomy import load, predict


def run(root, model_path, output):
    if output.exists():
        raise ValueError('Choose a new report path')
    torch.set_num_threads(4)
    model = load(model_path)
    cases, errors = [], []
    for patient in sorted((root/'Annotation').iterdir()):
        if not patient.is_dir():
            continue
        source = patient/'images/spine_image.dcm'
        mask_path = patient/'segmentations/spine_image.seg.nrrd'
        image = read_dxa(source)
        masks = read_seg_nrrd(mask_path, image.pixels.shape)
        prediction = predict(model, image.pixels)
        levels = []
        for level in range(1, 5):
            yy, xx = np.where(masks[f'Lumbar_{level}_bone_area'])
            if not len(xx):
                raise ValueError('Missing numbered public ROI')
            center = np.array([xx.mean(), yy.mean()])
            height = int(yy.max()-yy.min()+1)
            error = float(np.linalg.norm(prediction[level-1]-center)/height)
            errors.append(error)
            levels.append({'level': f'L{level}', 'reference_roi_center': center.tolist(),
                           'prediction': prediction[level-1].tolist(),
                           'roi_bbox_height': height, 'error_roi_heights': error})
        cases.append({'source_sha256': sha256(source), 'mask_sha256': sha256(mask_path), 'levels': levels})
    if not errors:
        raise ValueError('No numbered DXA references')
    report = {'clinical_validation': False, 'release_modified': False,
              'protocol': 'Frozen BUU model; untouched named DXA bone-area ROIs; numbering never reassigned from references',
              'model_sha256': sha256(model_path), 'code_sha256': sha256(Path(__file__)),
              'patients': len(cases), 'reference_bodies': len(errors),
              'within_half_roi_bbox_height': int((np.asarray(errors) <= .5).sum()),
              'numbered_roi_center_hit_rate': float((np.asarray(errors) <= .5).mean()),
              'median_error_roi_heights': float(np.median(errors)), 'cases': cases,
              'limitations': ['Bone-area ROI centers and bounding heights are not anatomical vertebral center/height ground truth.',
                              'No L5, Th12, iliac crest or disc reference.',
                              'Ten Hologic patients do not establish organizer GE DXA performance.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'limitations')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'model', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.model, args.output)
