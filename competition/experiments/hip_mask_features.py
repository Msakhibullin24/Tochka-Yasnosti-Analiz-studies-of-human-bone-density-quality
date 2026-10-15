"""Candidate hip measurements from TSXR masks and a synthetic landmark model.

No QC targets are read. Calibration and source/mask/checkpoint identities are
bound to output artifacts. Semantic reliability on DXA is not assumed.
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
from experiments.mask_anatomy_features import largest_component, mask_statistics
from experiments.train_silhouette_heatmaps import Heatmaps, normalize_proximal_mask, predict, coordinates, ORDER
from source_integrity import inspect_sources
from gpu_research.common import fingerprint


def hip_measurements(masks, pixels, sx, sy, model, device='cuda'):
    if masks.shape != (23, *pixels.shape):
        raise ValueError('Expected 23 independent musculoskeletal masks')
    femur = largest_component(masks[6] | masks[7])
    pelvis = largest_component(masks[8] | masks[9])
    features, centers = {}, {}
    for name, mask in [('femur', femur), ('pelvis', pelvis)]:
        stats, center = mask_statistics(mask, pixels, sx, sy)
        features.update({f'anat_{name}_{k}': v for k, v in stats.items()})
        centers[name] = center
    features['anat_femur_pelvis_overlap_fraction'] = float((femur & pelvis).sum() / max(1, femur.sum()))
    if (centers['femur'] is None or features.get('anat_femur_width_mm', 0) < 25. or
            features.get('anat_femur_height_mm', 0) < 70. or features.get('anat_femur_area_fraction', 0) < .05):
        features['anat_landmark_input_valid'] = 0.
        return features, None
    try:
        normalized, affine = normalize_proximal_mask(femur, sx, sy)
    except ValueError:
        features['anat_landmark_input_valid'] = 0.
        return features, None
    features['anat_landmark_input_valid'] = 1.
    if getattr(model, 'input_channels', 1) == 1:
        points = predict(model, normalized[None], device)[0]
    else:
        context = cv2.warpAffine(pelvis.astype(np.uint8), affine, (320, 320), flags=cv2.INTER_NEAREST)
        channels = np.stack([cv2.resize(m.astype(np.float32), (128, 128), interpolation=cv2.INTER_NEAREST)
                             for m in (normalized, context)])
        with torch.inference_mode():
            points, _ = coordinates(model(torch.tensor(channels[None], dtype=torch.float32, device=device)))
        points = (points[0].cpu().numpy() + 1) * 160 - .5
    inverse = cv2.invertAffineTransform(affine)
    points = points @ inverse[:, :2].T + inverse[:, 2]
    height, width = pixels.shape
    in_bounds = ((points >= 0) & (points < [width, height])).all(axis=1)
    for name, point, valid in zip(ORDER, points, in_bounds):
        features[f'anat_{name}_in_bounds'] = float(valid)
        features[f'anat_{name}_x'] = float(point[0] / width)
        features[f'anat_{name}_y'] = float(point[1] / height)
    axis = (points[2] - points[3]) * [sx, sy]
    features['anat_neck_axis_length_mm'] = float(np.linalg.norm(axis))
    features['anat_neck_axis_valid'] = float(np.linalg.norm(axis) >= 5.)
    if np.linalg.norm(axis) >= 5.:
        features['anat_neck_abs_angle_deg'] = float(abs(np.degrees(np.arctan2(axis[1], axis[0]))))
    features['anat_troch_top_margin_mm'] = float(points[0, 1] * sy)
    if centers['pelvis'] is not None:
        direction = np.sign(centers['pelvis'][0] - centers['femur'][0])
        yy, xx = np.where(femur)
        lateral_margin = xx.min() * sx if direction > 0 else (width - 1 - xx.max()) * sx
        features['anat_lateral_margin_mm'] = float(lateral_margin)
        py, _ = np.where(pelvis)
        features['anat_ischium_bottom_margin_mm'] = float((height - 1 - py.max()) * sy)
        # Compare a candidate lesser-trochanter point to the distal medial cortex.
        valid_rows, medial, shaft = [], [], []
        for y in range(int(np.quantile(yy, .7)), int(yy.max()) + 1):
            xs = np.flatnonzero(femur[y])
            if len(xs) >= 3:
                valid_rows.append(y)
                medial.append(xs.max() if direction > 0 else xs.min())
                shaft.append(xs.mean())
        if len(valid_rows) >= 5:
            a, b = np.polyfit(valid_rows, medial, 1)
            features['anat_lesser_protrusion_mm'] = float((points[1, 0] - (a * points[1, 1] + b)) * direction * sx)
            a, _ = np.polyfit(np.asarray(valid_rows) * sy, np.asarray(shaft) * sx, 1)
            features['anat_shaft_abs_angle_deg'] = float(abs(np.degrees(np.arctan(a))))
    return features, points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spine-features', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--mirror-probe', type=Path)
    parser.add_argument('--spine-pelvis-probe', type=Path)
    parser.add_argument('--landmarks', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    torch.set_num_threads(4)
    source = inspect_sources([('organiser', args.labels, args.dataset)])
    source_id = fingerprint(source['entries'])
    probe = json.loads((args.probe / 'probe.json').read_text())
    mirrored = json.loads((args.mirror_probe / 'probe.json').read_text()) if args.mirror_probe else None
    report = json.loads(args.spine_features.read_text())
    label_hash = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    if report['labels_sha256'] != label_hash or report['source_fingerprint'] != source_id or probe['labels_sha256'] != label_hash:
        raise ValueError('Source identity differs between anatomy branches')
    if mirrored is not None and (mirrored.get('labels_sha256') != label_hash or
            mirrored['checkpoint_sha256'] != probe['checkpoint_sha256'] or
            not mirrored.get('input_horizontally_flipped')):
        raise ValueError('Mirrored mask source identity differs')
    bundle = torch.load(args.landmarks, weights_only=True, map_location='cpu')
    if bundle.get('normalization') != 'physical_proximal_mask150mm_10mm_pad' or bundle['landmark_order'] != list(ORDER):
        raise ValueError('Unexpected landmark representation')
    model = Heatmaps(bundle.get('input_channels', 1)).cuda().eval()
    model.load_state_dict(bundle['state_dict'], strict=True)
    rows = pd.read_csv(args.labels)
    rows = rows[rows.quality_class.notna()].copy()
    if args.spine_pelvis_probe:
        spine_pelvis = json.loads((args.spine_pelvis_probe / 'probe.json').read_text())
        if spine_pelvis['labels_sha256'] != label_hash or spine_pelvis['checkpoint_sha256'] != probe['checkpoint_sha256']:
            raise ValueError('Spine pelvis checkpoint identity differs')
        pelvis_refs = {r['pixel_sha256']: r for r in spine_pelvis['results'] if r['polarity'] == 'bone-bright'}
        for row in rows.loc[rows.region.eq('spine')].itertuples():
            ref = pelvis_refs[row.pixel_sha256]
            path = args.dataset / row.first_source_path
            image = read_dxa(path)
            mask_path = args.spine_pelvis_probe / f'{row.pixel_sha256[:16]}-bone-bright.npy'
            if (image.pixel_sha256 != ref['decoded_pixel_sha256'] or
                    hashlib.sha256(path.read_bytes()).hexdigest() != ref['source_file_sha256'] or
                    hashlib.sha256(mask_path.read_bytes()).hexdigest() != ref['mask_sha256']):
                raise ValueError('Spine pelvis source identity differs')
            masks = np.load(mask_path)
            values = report['entries'][row.pixel_sha256]['features']
            for name, channel in [('iliac_left', 8), ('iliac_right', 9)]:
                stats, _ = mask_statistics(masks[channel], image.pixels, image.pixel_mm_x, image.pixel_mm)
                values.update({f'anat_{name}_{k}': v for k, v in stats.items()})
        report['columns_by_group']['spine'] = sorted({k for row in rows.loc[rows.region.eq('spine')].itertuples()
                                                    for k in report['entries'][row.pixel_sha256]['features']})
    references = {r['pixel_sha256']: r for r in probe['results'] if r['polarity'] == 'bone-bright'}
    mirror_references = {r['pixel_sha256']: r for r in mirrored['results'] if r['polarity'] == 'bone-bright'} if mirrored else {}
    args.output.mkdir(parents=True)
    tiles, candidate_points = [], []
    for row in rows.loc[rows.region.str.startswith('hip')].itertuples():
        reference = references[row.pixel_sha256]
        path = args.dataset / row.first_source_path
        image = read_dxa(path)
        mask_path = args.probe / f'{row.pixel_sha256[:16]}-bone-bright.npy'
        if (image.pixel_sha256 != reference['decoded_pixel_sha256'] or
                hashlib.sha256(path.read_bytes()).hexdigest() != reference['source_file_sha256'] or
                hashlib.sha256(mask_path.read_bytes()).hexdigest() != reference['mask_sha256']):
            raise ValueError('Source or mask identity differs')
        masks = np.load(mask_path)
        if mirrored is not None:
            mirror = mirror_references[row.pixel_sha256]
            mirror_path = args.mirror_probe / f'{row.pixel_sha256[:16]}-bone-bright.npy'
            if (mirror['decoded_pixel_sha256'] != image.pixel_sha256 or
                    mirror['source_file_sha256'] != reference['source_file_sha256'] or
                    hashlib.sha256(mirror_path.read_bytes()).hexdigest() != mirror['mask_sha256']):
                raise ValueError('Mirrored mask checksum differs')
            other = np.load(mirror_path)
            # Independent labels are unions across left/right: choosing by mask
            # area does not inspect QC targets or interchange patient labels.
            for pair in ((6, 7), (8, 9)):
                a, b = pair
                first, second = largest_component(masks[a] | masks[b]), largest_component(other[a] | other[b])
                if second.sum() > first.sum():
                    masks[a], masks[b] = other[a], other[b]
        features, points = hip_measurements(masks, image.pixels, image.pixel_mm_x, image.pixel_mm, model)
        report['entries'][row.pixel_sha256] = {'features': features}
        candidate_points.append({'historical_pixel_sha256': row.pixel_sha256,
                                 'points': points.tolist() if points is not None else None, 'clinical_validation': False})
        if len(tiles) < 20:
            canvas = cv2.cvtColor(image.pixels, cv2.COLOR_GRAY2BGR)
            if points is not None:
                for name, point in zip(('GT', 'LT', 'head', 'neck'), points):
                    p = tuple(np.rint(point).astype(int))
                    cv2.circle(canvas, p, 3, (0, 0, 255), -1)
                    cv2.putText(canvas, name, p, cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
            tiles.append(cv2.resize(canvas, (300, 300)))
    columns = sorted({k for row in rows.loc[rows.region.str.startswith('hip')].itertuples()
                      for k in report['entries'][row.pixel_sha256]['features']})
    report['columns_by_group']['hip'] = columns
    report['hip_checkpoint_sha256'] = probe['checkpoint_sha256']
    report['hip_mirror_area_selection'] = mirrored is not None
    report['spine_pelvis_masks_used'] = bool(args.spine_pelvis_probe)
    report['hip_landmark_checkpoint_sha256'] = hashlib.sha256(args.landmarks.read_bytes()).hexdigest()
    (args.output / 'features.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output / 'landmark_candidates.json').write_text(json.dumps(candidate_points, indent=2) + '\n')
    if tiles:
        cv2.imwrite(str(args.output / 'contact-sheet.png'), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]))
    print(json.dumps({'hip_images': len(candidate_points), 'hip_features': len(columns),
                      'landmarks_available': sum(p['points'] is not None for p in candidate_points)}), flush=True)


if __name__ == '__main__':
    main()
