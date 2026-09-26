"""Materialize labeled geometric perturbations from existing DXA ROI annotations."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from anatomy_training_data import records, load_record, split_records
from dxaqc.anatomical_roi import propose_roi, compare_roi
from dxaqc.dicom_io import read_dxa
from evaluate_external_roi_masks import read_seg_nrrd
from dxaqc.synthetic_defects import crop_pair, rotate_pair, artifact_pair, transform_polygon
from evaluate_organizer_dataset import sha256


def roi_displacements(masks, group, split, width, source_sha256, annotation_sha256):
    """Retain only fully in-frame shifts; measure geometry without QC labels."""
    cases = []
    for name, reference in masks.items():
        reference = np.asarray(reference, dtype=bool)
        if reference.ndim != 2 or reference.shape[1] != width or not reference.any():
            raise ValueError('Invalid ROI reference raster')
        proposal = propose_roi(reference, purpose=name, reference_origin='published_named_roi')
        original = compare_roi(proposal['polygon'], reference, purpose=name, reference_origin='published_named_roi')
        shifted = transform_polygon(proposal['polygon'], [[1., 0, 15.], [0, 1., 0]])
        if np.max(np.asarray(shifted)[:, 0]) >= width:
            continue
        altered = compare_roi(shifted, reference, purpose=name, reference_origin='published_named_roi')
        cases.append({'parent_group': group, 'split': split, 'purpose': name,
                      'original': original, 'shifted': altered, 'source_sha256': source_sha256,
                      'annotation_sha256': annotation_sha256, 'displacement_pixels': [15, 0],
                      'target_origin': 'synthetic_geometry', 'qc_targets': {}})
    return cases


def run(aasce, dxa, output):
    if output.exists():
        raise ValueError('Choose a new curriculum directory')
    rows, _ = records(aasce, dxa)
    test = split_records(rows)
    output.mkdir(parents=True)
    cases, roi_cases = [], []
    for i, row in enumerate(rows):
        if row['source'] != 'ramathibodi':
            continue
        pixels, target = load_record(row)
        masks = {f'L{k}': target == 12+k for k in range(1, 5)}
        h, w = pixels.shape
        parent = row['group'].split(':')[1][:12]
        split = 'test' if test[i] else 'train'
        variants = []
        for angle in (-10, 10):
            # DICOM spacing is read from the source, never copied from a PNG default.
            image = read_dxa(Path(row['image']))
            variants.append((f'rotate{angle:+d}', rotate_pair(pixels, masks, angle,
                              (image.pixel_mm_x or image.pixel_mm, image.pixel_mm))))
        variants.append(('crop', crop_pair(pixels, masks, (0, 0, w, max(1, round(h*.65))))))
        variants.append(('artifact', artifact_pair(pixels, masks, (w//3, h//3, w//3+8, h//3+25))))
        for kind, (altered, annotation, evidence) in variants:
            stem = f'{parent}_{kind}'
            cv2.imwrite(str(output/f'{stem}.png'), altered)
            np.savez_compressed(output/f'{stem}_masks.npz', **annotation)
            cases.append({'parent_group': row['group'], 'split': split, 'kind': kind,
                          'image': f'{stem}.png', 'masks': f'{stem}_masks.npz',
                          'source_sha256': row['image_sha256'], **evidence})
        roi_cases.extend(roi_displacements(masks, row['group'], split, w,
                                            row['image_sha256'], row['annotation_sha256']))
        hip_path = Path(row['image']).with_name('hip_image.dcm')
        hip_annotation = Path(row['annotation']).with_name('hip_image.seg.nrrd')
        hip = read_dxa(hip_path)
        hip_masks = read_seg_nrrd(hip_annotation, hip.pixels.shape)
        roi_cases.extend(roi_displacements({'femoral_neck_roi': hip_masks['Femoral_Neck_bone_area']},
                                            row['group'], split, hip.pixels.shape[1],
                                            sha256(hip_path), sha256(hip_annotation)))
    report = {'cases': cases, 'roi_cases': roi_cases, 'code_sha256': sha256(Path(__file__)),
              'clinical_validation': False, 'ground_truth_scope': 'controlled geometry only',
              'train_groups': sorted({c['parent_group'] for c in cases if c['split'] == 'train'}),
              'test_groups': sorted({c['parent_group'] for c in cases if c['split'] == 'test'})}
    if any(c['parent_group'] not in report[c['split']+'_groups'] for c in roi_cases):
        raise ValueError('ROI parent split differs from image curriculum')
    if set(report['train_groups']) & set(report['test_groups']):
        raise ValueError('Derived views leak across splits')
    (output/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'variants': len(cases), 'roi_pairs': len(roi_cases)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aasce', 'dxa', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); run(args.aasce, args.dxa, args.output)
