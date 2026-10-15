"""Measure candidate hip anatomy against ten existing publisher DXA ROI masks.

No mask/point matching or parameter fitting to the reference. Publisher vendor
ROIs are measurement regions, not a full anatomical landmark reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from evaluate_external_roi_masks import read_seg_nrrd, overlap, sha256
from experiments.hip_mask_features import hip_measurements
from experiments.mask_anatomy_features import largest_component
from experiments.train_silhouette_heatmaps import Heatmaps
from experiments.neck_roi_from_mask import neck_roi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'manifest', 'probe', 'mirror_probe', 'landmarks', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    torch.set_num_threads(4)
    bundle = torch.load(args.landmarks, map_location='cpu', weights_only=True)
    if bundle['normalization'] != 'physical_proximal_mask150mm_10mm_pad':
        raise ValueError('Unexpected model normalization')
    model = Heatmaps(bundle.get('input_channels', 1)).cuda().eval()
    model.load_state_dict(bundle['state_dict'], strict=True)
    metadata = [json.loads((p / 'probe.json').read_text()) for p in (args.probe, args.mirror_probe)]
    if any(p['labels_sha256'] != sha256(args.manifest) for p in metadata) or metadata[0]['checkpoint_sha256'] != metadata[1]['checkpoint_sha256']:
        raise ValueError('Probe checkpoint/source identities differ')
    rows = pd.read_csv(args.manifest)
    cases = []
    for row in rows.itertuples():
        path = args.dataset / row.first_source_path
        image = read_dxa(path)
        variants = []
        for directory, meta in zip((args.probe, args.mirror_probe), metadata):
            record = next(r for r in meta['results'] if r['pixel_sha256'] == row.pixel_sha256 and r['polarity'] == 'bone-bright')
            mask_path = directory / f'{row.pixel_sha256[:16]}-bone-bright.npy'
            if record['decoded_pixel_sha256'] != image.pixel_sha256 or record['source_file_sha256'] != sha256(path) or record['mask_sha256'] != sha256(mask_path):
                raise ValueError('Source/mask identity differs')
            variants.append(np.load(mask_path))
        masks = variants[0].copy()
        for a, b in ((6, 7), (8, 9)):
            if largest_component(variants[1][a] | variants[1][b]).sum() > largest_component(masks[a] | masks[b]).sum():
                masks[a], masks[b] = variants[1][a], variants[1][b]
        features, points = hip_measurements(masks, image.pixels, image.pixel_mm_x, image.pixel_mm, model)
        femur = largest_component(masks[6] | masks[7])
        reference_path = path.parent.parent / 'segmentations/hip_image.seg.nrrd'
        reference = read_seg_nrrd(reference_path, image.pixels.shape)
        bone, total = reference['Femoral_Neck_bone_area'], reference['Femoral_Total_bone_area']
        neck_footprint = bone | reference['Femoral_Neck_soft_tissue_area']
        proposal = {'status': 'unavailable', 'reason': 'landmarks_unavailable'}
        inside = None
        if points is not None:
            neck = np.rint(points[3]).astype(int)
            inside = bool(bone[neck[1], neck[0]]) if ((neck >= 0) & (neck < image.pixels.shape[::-1])).all() else False
            proposal = neck_roi(femur, points[2], points[3], image.pixel_mm_x, image.pixel_mm)
        comparison = None
        if proposal['status'] == 'proposed':
            raster = np.zeros(image.pixels.shape, np.uint8)
            cv2.fillConvexPoly(raster, np.rint(proposal['polygon']).astype(np.int32), 1)
            comparison = overlap(neck_footprint, raster)
        cases.append({'case_sha256': hashlib.sha256(path.parent.parent.name.encode()).hexdigest(),
                      'source_sha256': sha256(path), 'reference_sha256': sha256(reference_path),
                      'scale_source': image.pixel_mm_source, 'landmarks_available': points is not None,
                      'neck_center_inside_vendor_neck_bone_roi': inside,
                      'femur_mask_recall_of_vendor_total_bone_roi': float((femur & total).sum() / max(1, total.sum())),
                      'proposal': proposal, 'proposal_overlap_vendor_neck_roi': comparison})
    compared = [r['proposal_overlap_vendor_neck_roi']['dice'] for r in cases if r['proposal_overlap_vendor_neck_roi'] is not None]
    report = {'protocol': __doc__, 'images': len(cases), 'cases': cases,
              'source_url': 'https://data.mait.ai.in.th/dataset/bone-mineral-density-and-vertebral-fracture-asseessment-public-sharing',
              'landmarks_available': sum(c['landmarks_available'] for c in cases),
              'neck_center_inside_vendor_neck_bone_roi': sum(c['neck_center_inside_vendor_neck_bone_roi'] is True for c in cases),
              'mean_femur_recall_of_total_bone_roi': float(np.mean([c['femur_mask_recall_of_vendor_total_bone_roi'] for c in cases])),
              'roi_proposals_available': len(compared), 'mean_proposal_dice': float(np.mean(compared)) if compared else None,
              'model_sha256': sha256(args.landmarks), 'segmentation_checkpoint_sha256': metadata[0]['checkpoint_sha256'],
              'code_sha256': sha256(Path(__file__)), 'clinical_validation': False, 'automatic_promotions': 0,
              'limitations': ['Vendor total and neck ROIs do not annotate complete femur contours or trochanter points.',
                              'No new independent cohort; ten previously studied published Annotation folders.',
                              'device_default physical scale is assumed, not verified; no millimeter accuracy claim.',
                              'Proposed ROI dimensions are engineering parameters, not organizer-defined tolerances.']}
    args.output.mkdir(parents=True)
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'protocol', 'limitations')}), flush=True)


if __name__ == '__main__':
    main()
