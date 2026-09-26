"""Audit unnumbered spine body candidates against public named bone-area ROIs.

One-to-one matching measures localization only; it never grants a predicted body
an anatomical level. Vendor ROIs are weaker references than anatomical contours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from dxaqc.anatomy import detect_landmarks
from dxaqc.dicom_io import read_dxa
from dxaqc.geometry import measure_image
from evaluate_external_roi_masks import read_seg_nrrd
from evaluate_organizer_dataset import sha256


def match_centers(references, candidates):
    if not len(candidates) or not len(references):
        return []
    reference = np.asarray(references, dtype=float)
    candidate = np.asarray(candidates, dtype=float)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('Nonfinite body coordinates')
    distances = np.linalg.norm(reference[:, None, :] - candidate[None, :, :], axis=2)
    left, right = linear_sum_assignment(distances)
    return [{'reference_index': int(a), 'candidate_index': int(b),
             'distance_pixels': float(distances[a, b])} for a, b in zip(left, right)]


def audit(root: Path, method='released_midpoints'):
    if method not in ('released_midpoints', 'intensity_minima'):
        raise ValueError('Unknown anatomy candidate method')
    cases = []
    for patient in sorted((root/'Annotation').iterdir()):
        if not patient.is_dir():
            continue
        source = patient/'images/spine_image.dcm'
        mask_path = patient/'segmentations/spine_image.seg.nrrd'
        image = read_dxa(source)
        masks = read_seg_nrrd(mask_path, image.pixels.shape)
        names, centers = [], []
        for level in range(1, 5):
            name = f'Lumbar_{level}_bone_area'
            mask = masks[name]
            yy, xx = np.where(mask)
            if not len(xx):
                continue
            names.append(name); centers.append([float(xx.mean()), float(yy.mean())])
        measured = measure_image(image.pixels, 'spine', image.pixel_mm, image.pixel_mm_x)
        anatomy = detect_landmarks(image.pixels, 'spine', measured.overlay,
                                   image.pixel_mm, image.pixel_mm_x or image.pixel_mm)
        bodies = next(item['points'] for item in anatomy['landmarks']
                      if item['name']=='vertebral_body_candidates')
        if method == 'intensity_minima':
            from spine_body_candidate import detect
            bodies = detect(image.pixels, image.pixel_mm, image.pixel_mm_x or image.pixel_mm)['points']
        matched = match_centers(centers, bodies)
        for match in matched:
            match['reference_name'] = names[match['reference_index']]
            match['distance_fraction_of_image_height'] = match['distance_pixels']/image.pixels.shape[0]
        cases.append({'patient_sha256': hashlib.sha256(patient.name.encode()).hexdigest(),
                      'source_sha256': sha256(source), 'mask_sha256': sha256(mask_path),
                      'reference_centers': dict(zip(names, centers)), 'candidate_centers': bodies,
                      'matches': matched, 'unmatched_references': len(centers)-len(matched),
                      'numbering_predicted': False, 'clinical_validation': False})
    if not cases:
        raise ValueError('No source anatomy references')
    distances = [item['distance_pixels'] for case in cases for item in case['matches']]
    return {'scope': 'diagnostic localization against public bone-area ROIs; no reference or model modification',
            'method': method,
            'clinical_validation': False, 'patients': len(cases),
            'reference_bodies': sum(len(case['reference_centers']) for case in cases),
            'matched_bodies': len(distances),
            'unmatched_reference_bodies': sum(case['unmatched_references'] for case in cases),
            'median_center_distance_pixels': float(np.median(distances)) if distances else None,
            'max_center_distance_pixels': max(distances, default=None),
            'limitations': ['Bone-area ROIs are not complete vertebral contours.',
                            'Matching uses the reference and cannot establish predicted L1-L4 numbering.',
                            'No Th12 or iliac crest annotation exists in this ROI source.',
                            'Raster-size agreement does not attest all source coordinate transforms.',
                            'Hologic source exports differ from organizer GE images.'], 'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--method', choices=('released_midpoints', 'intensity_minima'), default='released_midpoints')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file')
    result = audit(args.root, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('cases', 'limitations')}))
