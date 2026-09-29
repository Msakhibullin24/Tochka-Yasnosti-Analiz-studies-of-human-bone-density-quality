"""Evaluate a reviewed offline package after checking its source provenance."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

import numpy as np

from review_geometry import axis_points, contour_mask
from amodal_visibility import projected_visibility

from dxaqc.anatomical_roi import polygon_mask
from dxaqc.mask_raster import decode_raster

from dxaqc.dicom_io import read_dxa
from dxaqc.neck_roi_geometry import neck_geometry
from dxaqc.validate_anatomy import evaluate, read_rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_review(results, source, reference, tolerance_mm=5.):
    results, source, reference = map(Path, (results, source, reference))
    source = source.resolve()
    labels = json.loads(reference.read_text())
    if labels.get('results_sha256') != digest(results) or labels.get('review_package_version') not in (1, 2, 3, 4):
        raise ValueError('Reference must match the reviewed results artifact and package version')
    rows = read_rows(results)
    indexed = {row['image_uid']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('Source image UIDs are ambiguous')
    for case in labels.get('cases', []):
        row = indexed.get(case.get('image_uid'))
        if row is None:
            raise ValueError('Reference image is absent from results')
        path = (source/row['path_to_file']).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError('Reference source path escapes source directory or is absent')
        if case.get('source_sha256') != digest(path):
            raise ValueError('Source DICOM differs from the reviewed image')
        image = read_dxa(path)
        height, width = image.pixels.shape
        if (case.get('pixel_sha256') != image.pixel_sha256 or image.image_uid != row['image_uid']
                or case.get('width') != width or case.get('height') != height
                or width != int(row['image_width']) or height != int(row['image_height'])):
            raise ValueError('Decoded source pixels, identity or geometry differ from the reference')
    report = evaluate(rows, labels, tolerance_mm)
    report.update(reference_sha256=digest(reference), results_sha256=digest(results),
                  source_provenance_verified=True, clinical_validation=False,
                  requirements_complete=False,
                  unassessed_requirements=['vertebral_numbering_and_roi', 'anatomical_neck_roi',
                                          'scan_coverage', 'rotation_subtypes'])
    if labels['review_package_version'] >= 2:
        report['named_regions'], report['rotation'] = evaluate_regions(rows, labels['cases'], labels['review_package_version'])
        report['unassessed_requirements'] = ['scan_coverage', 'rotation_subtype_prediction_accuracy',
                                              'disc_boundary_correctness', 'roi_clinical_validity']
    if labels['review_package_version'] >= 3:
        report['neck_axis'] = evaluate_axes(rows, labels['cases'])
        report['expert_neck_roi_geometry'] = evaluate_expert_neck_roi_geometry(rows, labels['cases'])
        report['unassessed_requirements'].extend(['th12_full_body_visibility', 'clinical_rotation_rule'])
    if labels['review_package_version'] >= 4:
        report['th12_full_body'] = evaluate_th12_full_body(rows, labels['cases'])
        report['unassessed_requirements'].append('th12_full_body_prediction_accuracy')
    return report


def evaluate_th12_full_body(rows, cases):
    """Record independently reviewed amodal geometry without a fabricated DXA verdict."""
    indexed = {r['image_uid']: r for r in rows}
    output = {'assessable': 0, 'unassessable': 0, 'cases': [],
              'prediction_accuracy': None, 'clinical_validation': False,
              'fraction_definition': 'projected area and vertical extent, separately; no clinical equivalence asserted'}
    for case in cases:
        if 'Th12' not in case['landmarks']:
            if 'th12_full_body' in case:
                raise ValueError('Hip case must not have Th12 full-body geometry')
            continue
        annotation = case.get('th12_full_body')
        if not isinstance(annotation, dict) or set(annotation) != {'status', 'quadrilateral'}:
            raise ValueError('Th12 full-body status and quadrilateral must be explicit')
        status, points = annotation['status'], annotation['quadrilateral']
        if status == 'unassessable':
            if points is not None:
                raise ValueError('Unassessable Th12 must not have a full-body quadrilateral')
            output['unassessable'] += 1
            output['cases'].append({'image_uid': case['image_uid'], 'status': status,
                                    'visible_projected_area_fraction': None,
                                    'visible_vertical_extent_fraction': None})
            continue
        if status != 'assessable':
            raise ValueError('Th12 full-body assessability must be explicit')
        row = indexed[case['image_uid']]
        result = projected_visibility(points, (int(row['image_height']), int(row['image_width'])))
        visible = case['named_regions']['Th12']['visible']
        if (visible is True and result['visible_projected_area_fraction'] == 0
                or visible is False and result['visible_projected_area_fraction'] > 0):
            raise ValueError('Full Th12 geometry conflicts with visible-contour annotation')
        output['assessable'] += 1
        output['cases'].append({'image_uid': case['image_uid'], 'status': status,
                                'visible_projected_area_fraction': result['visible_projected_area_fraction'],
                                'visible_vertical_extent_fraction': result['visible_vertical_extent_fraction'],
                                'full_projected_area_pixels_squared': result['full_projected_area_pixels_squared']})
    return output


def evaluate_regions(rows, cases, package_version=2):
    """Score each named mask separately, keeping missing predictions in the mean.

    Region names measure agreement with expert numbering, not generic bone overlap.
    Rotation subtype labels have no counterpart in the current runtime outputs.
    """
    indexed = {r['image_uid']: r for r in rows}
    metrics, rotations = {}, Counter()
    for case in cases:
        row = indexed[case['image_uid']]
        spine = 'Th12' in case['landmarks']
        expected = ('L1', 'L2', 'L3', 'L4') if spine else ('femoral_neck_roi',)
        if package_version >= 3:
            expected = ('Th12', 'L1', 'L2', 'L3', 'L4') if spine else (
                'femoral_neck_roi', 'femur', 'greater_trochanter', 'lesser_trochanter', 'ischium')
        annotations = case.get('named_regions')
        if not isinstance(annotations, dict) or set(annotations) != set(expected):
            raise ValueError('All required named regions must be explicitly annotated')
        if not spine:
            rotation = case.get('rotation')
            if rotation not in ('none', 'insufficient', 'excessive', 'not_assessable'):
                raise ValueError('Hip rotation must have an explicit expert label')
            rotations[rotation] += 1
        shape = (int(row['image_height']), int(row['image_width']))
        learned = json.loads(row.get('learned_anatomy') or '{}')
        regions = learned.get('regions', []) if learned.get('status') == 'evaluated' else []
        if len({r['name'] for r in regions}) != len(regions):
            raise ValueError('Ambiguous named mask predictions')
        predictions = {r['name']: r for r in regions}
        for name in expected:
            annotation = annotations[name]
            if annotation.get('visible') is not True and annotation.get('visible') is not False:
                raise ValueError('Region visibility must be explicitly labelled')
            metric = metrics.setdefault(name, {'visible_references': 0, 'missing_predictions': 0,
                                               'invisible_references': 0, 'false_positive_on_invisible': 0,
                                               'dice_per_visible_reference': [], 'iou_per_visible_reference': []})
            candidate = predictions.get(name)
            if not annotation['visible']:
                if annotation.get('polygon') is not None:
                    raise ValueError('Invisible region must not have a contour')
                metric['invisible_references'] += 1
                metric['false_positive_on_invisible'] += int(candidate is not None)
                continue
            rasterize = contour_mask if package_version >= 3 and name != 'femoral_neck_roi' else polygon_mask
            reference = rasterize(annotation.get('polygon'), shape)
            metric['visible_references'] += 1
            if candidate is None:
                metric['missing_predictions'] += 1
                dice = iou = 0.
            else:
                prediction = (decode_raster(candidate['raster'], shape) if 'raster' in candidate
                              else rasterize(candidate['polygon'], shape))
                intersection = int((prediction & reference).sum())
                union = int((prediction | reference).sum())
                dice = 2 * intersection / int(prediction.sum() + reference.sum())
                iou = intersection / union
            metric['dice_per_visible_reference'].append(dice)
            metric['iou_per_visible_reference'].append(iou)
    for metric in metrics.values():
        for key in ('dice', 'iou'):
            values = metric[f'{key}_per_visible_reference']
            metric[f'mean_{key}_including_missing'] = float(np.mean(values)) if values else None
    return metrics, {'expert_label_counts': dict(rotations), 'prediction_accuracy': None,
                     'reason': 'Runtime does not predict insufficient/excessive rotation separately'}


def evaluate_axes(rows, cases):
    indexed = {r['image_uid']: r for r in rows}
    metric = {'visible_references': 0, 'missing_predictions': 0, 'unscaled_predictions': 0,
              'invisible_references': 0, 'false_positive_on_invisible': 0,
              'angle_errors_deg': [], 'endpoint_errors_mm': []}
    for case in cases:
        if 'Th12' in case['landmarks']:
            continue
        axes = case.get('axes')
        if not isinstance(axes, dict) or set(axes) != {'femoral_neck_axis'}:
            raise ValueError('Independent neck axis must be explicitly annotated')
        annotation = axes['femoral_neck_axis']
        row = indexed[case['image_uid']]
        shape = int(row['image_height']), int(row['image_width'])
        learned = json.loads(row.get('learned_anatomy') or '{}')
        candidates = [r for r in learned.get('regions', []) if r.get('name') == 'femoral_neck_axis'] if learned.get('status') == 'evaluated' else []
        if len(candidates) > 1:
            raise ValueError('Ambiguous neck axis prediction')
        candidate = candidates[0] if candidates else None
        if annotation.get('visible') is False:
            if annotation.get('points') is not None:
                raise ValueError('Invisible axis must not have endpoints')
            metric['invisible_references'] += 1
            metric['false_positive_on_invisible'] += int(candidate is not None)
            continue
        if annotation.get('visible') is not True:
            raise ValueError('Axis visibility must be explicitly labelled')
        truth = axis_points(annotation.get('points'), shape)
        metric['visible_references'] += 1
        if candidate is None:
            metric['missing_predictions'] += 1
            continue
        predicted = axis_points(candidate.get('points'), shape)
        sx, sy = float(row['pixel_mm_x']), float(row['pixel_mm'])
        if (not np.isfinite([sx, sy]).all() or min(sx, sy) <= 0
                or row.get('pixel_mm_source') in ('device_default', '', None)):
            metric['unscaled_predictions'] += 1
            continue
        truth, predicted = truth * [sx, sy], predicted * [sx, sy]
        a, b = truth[1]-truth[0], predicted[1]-predicted[0]
        angle = np.degrees(np.arccos(np.clip(abs(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)), 0, 1)))
        # Axis has no endpoint direction; choose the equivalent endpoint ordering.
        endpoint_error = min(np.linalg.norm(predicted-truth, axis=1).mean(),
                             np.linalg.norm(predicted[::-1]-truth, axis=1).mean())
        metric['angle_errors_deg'].append(float(angle))
        metric['endpoint_errors_mm'].append(float(endpoint_error))
    for field in ('angle_errors_deg', 'endpoint_errors_mm'):
        metric['mean_'+field] = float(np.mean(metric[field])) if metric[field] else None
    metric['prediction_coverage'] = (metric['visible_references']-metric['missing_predictions'])/metric['visible_references'] if metric['visible_references'] else None
    metric['scope'] = 'Agreement with an independent undirected neck axis; angular errors require reliable anisotropic physical scale'
    return metric


def evaluate_expert_neck_roi_geometry(rows, cases):
    """Measure reference ROI relationships without inventing a clinical threshold."""
    indexed = {row['image_uid']: row for row in rows}
    output = {'roi_cases': 0, 'measured': 0, 'partial': 0, 'unassessable': 0, 'unscaled': 0,
              'mean_deviation_from_perpendicular_deg': None, 'cases': [],
              'clinical_verdict': 'undetermined', 'clinical_validation': False,
              'scope': 'Expert ROI, femur, trochanter, ischium and independent neck axis; descriptive geometry only'}
    deviations = []
    for case in cases:
        if 'Th12' in case['landmarks']:
            continue
        output['roi_cases'] += 1
        row = indexed[case['image_uid']]
        shape = (int(row['image_height']), int(row['image_width']))
        regions = case['named_regions']
        axis = case['axes']['femoral_neck_axis']
        required = ('femoral_neck_roi', 'femur')
        if any(regions[name]['visible'] is not True for name in required) or axis['visible'] is not True:
            output['unassessable'] += 1
            output['cases'].append({'image_uid': case['image_uid'], 'status': 'unassessable',
                                    'reason': 'Reference ROI, femur or independent neck axis unavailable'})
            continue
        roi = polygon_mask(regions['femoral_neck_roi']['polygon'], shape)
        femur = contour_mask(regions['femur']['polygon'], shape)
        points = axis_points(axis['points'], shape)
        sx, sy = float(row['pixel_mm_x']), float(row['pixel_mm'])
        if (not np.isfinite([sx, sy]).all() or min(sx, sy) <= 0
                or row.get('pixel_mm_source') in ('device_default', '', None)):
            output['unscaled'] += 1
            output['cases'].append({'image_uid': case['image_uid'], 'status': 'unscaled',
                                    'reason': 'Reliable physical pixel scale unavailable'})
            continue
        measurement = neck_geometry(roi, femur, sx, sy, points)
        overlaps = {}
        for name in ('greater_trochanter', 'ischium'):
            annotation = regions[name]
            overlaps[name] = (int((roi & contour_mask(annotation['polygon'], shape)).sum())
                              if annotation['visible'] is True else None)
        measurement['reference_structure_overlap_pixels'] = overlaps
        measurement['image_uid'] = case['image_uid']
        output['measured' if measurement['status'] == 'measured' else 'partial'] += 1
        output['cases'].append(measurement)
        deviation = measurement['deviation_from_perpendicular_deg']
        if deviation is not None:
            deviations.append(deviation)
    if deviations:
        output['mean_deviation_from_perpendicular_deg'] = float(np.mean(deviations))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'source', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--tolerance-mm', type=float, default=5.)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('Choose a new output report')
        report = evaluate_review(args.results, args.source, args.reference, args.tolerance_mm)
        with args.output.open('x') as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'evaluated_cases': report['evaluated_cases'],
                          'reference_coverage': report['reference_coverage'],
                          'requirements_complete': False}))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f'REVIEW_VALIDATION_FAILED: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
