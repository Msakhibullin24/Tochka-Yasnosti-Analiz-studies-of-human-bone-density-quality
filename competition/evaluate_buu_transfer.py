"""Frozen projection and unnumbered localization checks on new BUU radiographs.

No model is fitted and no threshold is selected on this cohort. BUU annotations
are used only after inference. This evaluates transfer, not clinical DXA QC.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment

from dxaqc.anatomy import detect_landmarks
from dxaqc.embedding import WEIGHTS_SHA256
from evaluate_organizer_dataset import binary_metrics, sha256
from evaluate_projection_candidate import assessment, features
from spine_body_candidate import detect


def references(path: Path, width: int, height: int):
    rows = np.asarray(list(csv.reader(path.open())), dtype=float)
    if rows.shape != (10, 5) or not np.isfinite(rows).all():
        raise ValueError('AP annotations must contain ten finite five-column edges')
    endpoints = rows[:, :4].reshape(5, 2, 2, 2)
    if ((endpoints[..., 0] < 0).any() or (endpoints[..., 0] >= width).any()
            or (endpoints[..., 1] < 0).any() or (endpoints[..., 1] >= height).any()):
        raise ValueError('Reference endpoints outside source raster')
    centers = endpoints.mean(axis=(1, 2))
    body_heights = np.abs(endpoints[:, 1, :, 1].mean(axis=1)
                          - endpoints[:, 0, :, 1].mean(axis=1))
    if (body_heights <= 0).any() or not np.all(np.diff(centers[:, 1]) > 0):
        raise ValueError('Invalid or unordered reference vertebral edges')
    return centers, body_heights


def match_bodies(centers, heights, candidates):
    """Maximum-cardinality one-to-one matching inside half a reference height.

    Dummy assignments penalize missed bodies, preventing a far-away candidate
    from being credited as a localized vertebra. Distance has no physical unit.
    """
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    heights = np.asarray(heights, dtype=float)
    points = np.asarray(candidates, dtype=float).reshape(-1, 2)
    if (len(heights) != len(centers) or not np.isfinite(centers).all()
            or not np.isfinite(points).all() or not np.isfinite(heights).all()
            or (heights <= 0).any()):
        raise ValueError('Invalid matching coordinates or heights')
    n, m = len(centers), len(points)
    if not n or not m:
        return {'matched': [], 'missed': n, 'extra': m}
    distances = np.linalg.norm(centers[:, None] - points[None], axis=2)
    normalized = distances / heights[:, None]
    penalty = n + m + 1
    cost = np.zeros((n + m, n + m))
    cost[:n, :m] = np.where(normalized <= .5, normalized, penalty * 3)
    cost[:n, m:] = penalty
    cost[n:, :m] = penalty
    a, b = linear_sum_assignment(cost)
    matched = [{'reference_index': int(i), 'candidate_index': int(k),
                'distance_pixels': float(distances[i, k]),
                'distance_body_heights': float(normalized[i, k])}
               for i, k in zip(a, b) if i < n and k < m and normalized[i, k] <= .5]
    return {'matched': matched, 'missed': n-len(matched), 'extra': m-len(matched)}


def run(root: Path, models: Path, output: Path):
    import torch
    torch.set_num_threads(4)
    if output.exists():
        raise ValueError('Choose a new output path')
    frozen = {}
    for name in ('projection', 'projection_adapted'):
        path = models / (name + '.joblib')
        bundle = joblib.load(path)
        if (bundle.get('encoder_sha256') != WEIGHTS_SHA256 or bundle.get('status') != 'research_only'
                or bundle.get('classes') != ['frontal', 'lateral']
                or not np.array_equal(bundle['model'].classes_, [0, 1])):
            raise ValueError('Incompatible frozen projection candidate')
        frozen[name] = (bundle['model'], sha256(path))
    cases = []
    paths = sorted(root.rglob('*.jpg'))
    if not paths:
        raise ValueError('No BUU images')
    for number, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or path.stem[-1] not in ('0', '1'):
            raise ValueError(f'Invalid BUU image: {path}')
        lateral = int(path.stem[-1])
        vectors = features(image)
        case = {'file': str(path.relative_to(root)), 'patient_id': path.stem[:4],
                'source_sha256': sha256(path),
                'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'lateral_reference': lateral,
                'label_basis': 'BUU source AP/LA naming convention', 'projection': {}}
        for name, (model, _) in frozen.items():
            case['projection'][name] = assessment(float(model.predict_proba(vectors)[:, 1].mean()))
        if not lateral:
            h, w = image.shape
            centers, heights = references(path.with_suffix('.csv'), w, h)
            # Native rasters are used without source spacing assumptions.
            # Coordinates and errors are pixels, never fabricated millimetres.
            native = detect_landmarks(image, 'spine', {})
            midpoints = next(x['points'] for x in native['landmarks'] if x['name'] == 'vertebral_body_candidates')
            minima = detect(image, 1., 1.)['points']
            case['localization'] = {name: match_bodies(centers, heights, points)
                                    for name, points in [('released_midpoints', midpoints), ('intensity_minima', minima)]}
            case['reference_centers'] = centers.tolist()
            case['reference_heights'] = heights.tolist()
            case['annotation_sha256'] = sha256(path.with_suffix('.csv'))
        cases.append(case)
        if (number + 1) % 50 == 0:
            print(f'Evaluated {number+1}/{len(paths)}', flush=True)
    y = np.array([r['lateral_reference'] for r in cases])
    projection = {}
    for name, (_, digest) in frozen.items():
        scores = np.array([r['projection'][name]['lateral_score'] for r in cases])
        decisions = [r['projection'][name]['value'] for r in cases]
        accepted = np.array([d != 'unknown' for d in decisions])
        projection[name] = {'model_sha256': digest, 'binary_threshold': .5,
                            'binary_metrics': binary_metrics(y, (scores >= .5).astype(int), scores),
                            'abstentions': int((~accepted).sum()),
                            'accepted_errors': int(((scores >= .5) != y)[accepted].sum())}
    localization = {}
    for name in ('released_midpoints', 'intensity_minima'):
        records = [r['localization'][name] for r in cases if 'localization' in r]
        matches = [m for r in records for m in r['matched']]
        tp, fn, fp = len(matches), sum(r['missed'] for r in records), sum(r['extra'] for r in records)
        localization[name] = {'matched': tp, 'missed': fn, 'extra_candidates': fp,
                              'recall': tp/(tp+fn), 'precision': tp/(tp+fp) if tp+fp else 0.,
                              'median_matched_error_body_heights': float(np.median([m['distance_body_heights'] for m in matches])) if matches else None}
    report = {'schema_version': 1, 'source': 'https://services.informatics.buu.ac.th/spine/',
              'images': len(cases), 'patients': len({r['patient_id'] for r in cases}),
              'unique_pixels': len({r['pixel_sha256'] for r in cases}),
              'projection': projection, 'localization': localization,
              'evaluation_code_sha256': sha256(Path(__file__)),
              'clinical_dxa_validation': False, 'release_modified': False,
              'protocol': 'Frozen models; no fitting or threshold selection on BUU. Localization AP only, gated one-to-one matching <= half reference body height. No absolute numbering credit.',
              'limitations': ['Plain radiographs differ from DXA.', 'Source view labels are not new physician review.',
                              'Body edge centers are not full segmentation masks.', 'No physical pixel spacing supplied.',
                              'Th12, iliac crests and original DXA ROI are not labelled.'], 'cases': cases}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: report[k] for k in ('images', 'projection', 'localization')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'models', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.models, args.output)
