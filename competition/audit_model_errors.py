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
from dxaqc.model import CRITERIA, VIOLATION_LABEL, group_of


def audit(oof_path, labels_path, dataset, output):
    output.mkdir(parents=True, exist_ok=False)
    with labels_path.open() as stream:
        labels = {r['first_source_path']: r for r in csv.DictReader(stream) if r['quality_class']}
    with oof_path.open() as stream:
        predictions = list(csv.DictReader(stream))
    cases, cards = [], []
    for row in predictions:
        label = labels[row['source_path']]
        untyped = row['quality_pred'] == '1' and not row['violation_type']
        missed_axis = label.get('spine_axis') in ('1', '1.0') and VIOLATION_LABEL['spine_axis'] not in row['violation_type'].split(';')
        if not (untyped or missed_axis):
            continue
        image = read_dxa(dataset / row['source_path'])
        region = row['predicted_region']
        pixels = np.ascontiguousarray(image.pixels[:, ::-1]) if region == 'hip_left' else image.pixels
        measurement = measure_image(pixels, region, image.pixel_mm, image.pixel_mm_x)
        targets = [k for k in CRITERIA[group_of(label['region'])] if label.get(k) in ('1','1.0')]
        codes = [k for k in CRITERIA[group_of(region)] if VIOLATION_LABEL[k] in row['violation_type'].split(';')]
        item = {'source_path':row['source_path'], 'fold':int(row['fold']), 'repeat':int(row['repeat']),
                'review_reasons': [name for name, active in [('untyped_positive',untyped),('missed_axis',missed_axis)] if active],
                'reference_criteria':targets, 'reference_quality':int(float(label['quality_class'])),
                'predicted_type':row['violation_type'], 'predicted_quality':int(row['quality_pred']),
                'physical_axis_angle_deg':measurement.features.get('spine_angle_deg'),
                'curve_rms_mm':measurement.features.get('spine_curve_rms_mm'),
                'source_pixel_sha256':image.pixel_sha256}
        cases.append(item)
        bgr = render_overlay(image.pixels, region, measurement.overlay, measurement.features, codes,
                             int(row['quality_pred']), float(row['quality_score']), image.pixel_mm, image.pixel_mm_x)
        ok, encoded = cv2.imencode('.png', bgr)
        if not ok:
            raise RuntimeError('cannot encode review image')
        url = base64.b64encode(encoded.tobytes()).decode()
        cards.append('<article><h2>'+escape(row['source_path'])+'</h2><pre>'+escape(json.dumps(item,ensure_ascii=False,indent=2))+'</pre><img src="data:image/png;base64,'+url+'" alt="Измеренная ось и признаки; локализация не подтверждена"></article>')
    result={'scope':'diagnostic review of already inspected internal OOF; not a new independent evaluation',
            'oof_sha256':hashlib.sha256(oof_path.read_bytes()).hexdigest(),
            'labels_sha256':hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            'count':len(cases),'cases':cases}
    (output/'cases.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (output/'review.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Разбор ошибок</title><style>body{font:16px system-ui;margin:24px}article{border-top:2px solid #777;padding:20px}img{max-width:100%;max-height:1000px}h2,pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Ошибки оси и неопределённые типы</h1><p>Исходные метки не изменены. Изображение показано в физических пропорциях. Линия и ориентиры требуют экспертной проверки. Этот разбор не является новой независимой оценкой.</p>'+''.join(cards)+'</html>')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('oof','labels','dataset','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps({'cases':audit(args.oof,args.labels,args.dataset,args.output)['count']}))
