"""Create a review packet from existing held-out predictions; never fit or change labels."""
import argparse
import base64
import csv
import hashlib
import json
from html import escape
from pathlib import Path

import cv2
import numpy as np

from dxaqc.dicom_io import read_dxa
from dxaqc.geometry import measure_image
from dxaqc.explain import render_overlay
from dxaqc.anatomy import detect_landmarks, display_overlay
from dxaqc.model import CRITERIA, VIOLATION_LABEL, group_of


def review_reasons(row, label):
    """Select disagreements against existing labels without treating candidate points as truth."""
    predicted_types = row['violation_type'].split(';')
    reasons = []
    if row['quality_pred'] == '1' and not row['violation_type']:
        reasons.append('untyped_positive')
    if label.get('spine_axis') in ('1', '1.0') and VIOLATION_LABEL['spine_axis'] not in predicted_types:
        reasons.append('missed_axis')
    if label['region'].startswith('hip'):
        actual = label.get('hip_position_rotation') in ('1', '1.0')
        predicted = VIOLATION_LABEL['hip_position_rotation'] in predicted_types
        if actual and not predicted:
            reasons.append('missed_hip_position_rotation')
        elif not actual and predicted and label.get('hip_position_rotation') in ('0', '0.0'):
            reasons.append('false_hip_position_rotation')
    return reasons


def audit(oof_path, labels_path, dataset, output, repeat=0):
    with labels_path.open() as stream:
        labels = {r['first_source_path']: r for r in csv.DictReader(stream) if r['quality_class']}
    with oof_path.open() as stream:
        predictions = list(csv.DictReader(stream))
    cases, cards = [], []
    selected = [row for row in predictions if int(row['repeat']) == repeat]
    if not selected:
        raise ValueError(f'OOF has no predictions for repeat {repeat}')
    if len({row['source_path'] for row in selected}) != len(selected):
        raise ValueError('OOF contains duplicate source paths within one repeat')
    output.mkdir(parents=True, exist_ok=False)
    for row in selected:
        label = labels[row['source_path']]
        reasons = review_reasons(row, label)
        if not reasons:
            continue
        image = read_dxa(dataset / row['source_path'])
        region = row['predicted_region']
        pixels = np.ascontiguousarray(image.pixels[:, ::-1]) if region == 'hip_left' else image.pixels
        measurement = measure_image(pixels, region, image.pixel_mm, image.pixel_mm_x)
        landmarks = detect_landmarks(image.pixels, region, measurement.overlay,
                                     image.pixel_mm, image.pixel_mm_x)
        targets = [k for k in CRITERIA[group_of(label['region'])] if label.get(k) in ('1','1.0')]
        codes = [k for k in CRITERIA[group_of(region)] if VIOLATION_LABEL[k] in row['violation_type'].split(';')]
        item = {'source_path':row['source_path'], 'fold':int(row['fold']), 'repeat':int(row['repeat']),
                'review_reasons': reasons,
                'reference_criteria':targets, 'reference_quality':int(float(label['quality_class'])),
                'predicted_type':row['violation_type'], 'predicted_quality':int(row['quality_pred']),
                'quality_score':float(row['quality_score']),
                'physical_axis_angle_deg':measurement.features.get('spine_angle_deg'),
                'curve_rms_mm':measurement.features.get('spine_curve_rms_mm'),
                'hip_measurements': {key: (None if not np.isfinite(value) else float(value))
                                     for key, value in measurement.features.items()
                                     if region.startswith('hip') and key in
                                     {'shaft_angle_deg', 'shaft_width_mm', 'lesser_troch_protrusion_mm',
                                      'lesser_troch_notch_mm', 'medial_concavity_mm', 'troch_top_margin_mm',
                                      'ischium_bottom_margin_mm', 'lateral_margin_mm'}},
                'landmark_candidates': {point['name']: point['status'] for point in landmarks['landmarks']},
                'landmarks_clinically_validated': False,
                'source_pixel_sha256':image.pixel_sha256}
        cases.append(item)
        inspected_overlay = display_overlay({'region': region, 'overlay': measurement.overlay,
                                             'anatomy_candidates': landmarks}, image.pixels.shape[1])
        bgr = render_overlay(image.pixels, region, inspected_overlay, measurement.features, codes,
                             int(row['quality_pred']), float(row['quality_score']), image.pixel_mm, image.pixel_mm_x)
        ok, encoded = cv2.imencode('.png', bgr)
        if not ok:
            raise RuntimeError('cannot encode review image')
        url = base64.b64encode(encoded.tobytes()).decode()
        cards.append('<article><h2>'+escape(row['source_path'])+'</h2><pre>'+escape(json.dumps(item,ensure_ascii=False,indent=2))+'</pre><img src="data:image/png;base64,'+url+'" alt="Измеренная ось и признаки; локализация не подтверждена"></article>')
    counts = {reason: sum(reason in item['review_reasons'] for item in cases)
              for reason in ('untyped_positive', 'missed_axis', 'missed_hip_position_rotation',
                             'false_hip_position_rotation')}
    result={'scope':'diagnostic review of already inspected internal OOF; not a new independent evaluation',
            'oof_sha256':hashlib.sha256(oof_path.read_bytes()).hexdigest(),
            'labels_sha256':hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            'repeat':repeat, 'evaluated_images':len(selected), 'count':len(cases),
            'reason_counts':counts, 'cases':cases}
    hip_misses = [item for item in cases if 'missed_hip_position_rotation' in item['review_reasons']]
    hip_false = [item for item in cases if 'false_hip_position_rotation' in item['review_reasons']]
    summary = {key: value for key, value in result.items() if key != 'cases'}
    summary['hip_position_rotation'] = {
        'missed_with_binary_negative': sum(item['predicted_quality'] == 0 for item in hip_misses),
        'missed_with_binary_positive': sum(item['predicted_quality'] == 1 for item in hip_misses),
        'candidate_counts': {
            kind: {name: sum(item['landmark_candidates'].get(name) == 'candidate' for item in group)
                   for name in ('greater_trochanter', 'lesser_trochanter', 'femoral_neck', 'ischium')}
            for kind, group in (('missed', hip_misses), ('false_alarm', hip_false))},
    }
    (output/'cases.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (output/'review.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Разбор ошибок</title><style>body{font:16px system-ui;margin:24px}article{border-top:2px solid #777;padding:20px}img{max-width:100%;max-height:1000px}h2,pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Разбор ошибок критериев</h1><p>Исходные метки не изменены. Изображение показано в физических пропорциях. Линия и ориентиры требуют экспертной проверки. Этот разбор не является новой независимой оценкой.</p>'+''.join(cards)+'</html>')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('oof','labels','dataset','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--repeat', type=int, default=0,
                        help='One held-out repeat only, to avoid counting an image several times')
    args=parser.parse_args()
    result = audit(args.oof,args.labels,args.dataset,args.output,args.repeat)
    print(json.dumps({'cases':result['count'], 'reason_counts':result['reason_counts']}))
