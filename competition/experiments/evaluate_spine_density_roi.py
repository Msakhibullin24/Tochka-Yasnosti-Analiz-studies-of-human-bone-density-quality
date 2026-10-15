"""Compare density-split spine ROI proposals to existing publisher measurement ROIs.

ROI placement uses only pixels and named pretrained masks. References are read
only after proposals exist. The vendor ROI is not a disc-boundary reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from evaluate_external_roi_masks import read_seg_nrrd, overlap, sha256
from experiments.numbered_spine_roi import propose_spine_rois


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(4)
    predictor = nnUNetPredictor(use_mirroring=False, allow_tqdm=False, verbose=False, device=torch.device('cuda'))
    predictor.initialize_from_trained_model_folder(str(args.model), use_folds=(0,), checkpoint_name='checkpoint_best.pth')
    args.output.mkdir(parents=True)
    results = []
    for folder in sorted(args.dataset.iterdir()):
        path = folder / 'images/spine_image.dcm'
        if not path.is_file():
            continue
        image = read_dxa(path)
        masks = np.asarray(predictor.predict_single_npy_array(image.pixels[None, None].astype(np.float32),
            {'spacing': (999 * max(image.pixel_mm_x, image.pixel_mm), image.pixel_mm, image.pixel_mm_x)})).squeeze()
        proposals = propose_spine_rois(masks, image.pixels, image.pixel_mm_x, image.pixel_mm)
        reference_path = folder / 'segmentations/spine_image.seg.nrrd'
        reference = read_seg_nrrd(reference_path, image.pixels.shape)
        comparisons = []
        for proposal in proposals['proposals']:
            i = int(proposal['purpose'][1:])
            raster = np.zeros(image.pixels.shape, np.uint8)
            cv2.fillConvexPoly(raster, np.rint(proposal['polygon']).astype(np.int32), 1)
            bone = reference[f'Lumbar_{i}_bone_area']
            footprint = bone | reference[f'Lumbar_{i}_soft_tissue_area']
            comparisons.append({'level': proposal['purpose'], 'vendor_bone_overlap': overlap(bone, raster),
                                'vendor_footprint_overlap': overlap(footprint, raster),
                                'bone_roi_recall': float((bone & (raster > 0)).sum() / max(1, bone.sum()))})
        case_sha = hashlib.sha256(folder.name.encode()).hexdigest()
        results.append({'case_sha256': case_sha, 'source_sha256': sha256(path),
                        'reference_sha256': sha256(reference_path), 'scale_source': image.pixel_mm_source,
                        'proposal': proposals, 'comparisons': comparisons})
        np.save(args.output / f'{case_sha[:16]}-masks.npy', masks)
    comparisons = [c for r in results for c in r['comparisons']]
    if not results:
        raise ValueError('No existing annotated DXA folders')
    report = {'protocol': __doc__, 'images': len(results), 'cases': results,
              'levels_with_proposals': len(comparisons),
              'mean_vendor_bone_dice': float(np.mean([c['vendor_bone_overlap']['dice'] for c in comparisons])) if comparisons else None,
              'mean_vendor_footprint_dice': float(np.mean([c['vendor_footprint_overlap']['dice'] for c in comparisons])) if comparisons else None,
              'mean_vendor_bone_recall': float(np.mean([c['bone_roi_recall'] for c in comparisons])) if comparisons else None,
              'model_sha256': sha256(args.model / 'fold_0/checkpoint_best.pth'), 'code_sha256': sha256(Path(__file__)),
              'geometry_code_sha256': sha256(Path(__file__).with_name('numbered_spine_roi.py')),
              'clinical_validation': False, 'automatic_promotions': 0,
              'limitations': ['Ten previously inspected Hologic annotation folders; not a new cohort.',
                              'Vendor measurement ROIs differ from whole vertebral masks and have no disc reference.',
                              'Default reader physical scale; no millimeter error claim.',
                              'Density markers are proposals; unavailable outer neighbor prevents inventing that ROI.']}
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'protocol', 'limitations')}), flush=True)


if __name__ == '__main__':
    main()
