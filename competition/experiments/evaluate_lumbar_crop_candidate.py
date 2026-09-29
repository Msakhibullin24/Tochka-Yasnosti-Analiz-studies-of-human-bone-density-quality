"""Paired exploratory comparison against the deployed spine head, without promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anatomy_training_data import load_record
from train_anatomy_masks import metrics
from dxaqc.learned_anatomy import LearnedAnatomy, spatial_features, decode_masks, mask_regions, numbered_axis
from dxaqc.dicom_io import read_dxa
from audit_numbered_spine_axis import score


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.output.exists():
        raise ValueError('Choose a new report')
    torch.set_num_threads(4)
    models = {'deployed': LearnedAnatomy(args.deployed),
              'candidate': LearnedAnatomy(args.training / 'anatomy_masks.pt')}
    inventory = json.loads((args.training / 'inventory.json').read_text())['rows']
    cache = np.load(args.training / 'features.npy', mmap_mode='r')
    if cache.shape != (len(inventory), 896, 40, 40):
        raise ValueError('Unexpected feature cache shape')
    train_groups = {r['group'] for r in inventory if r['split'] == 'train'}
    test_groups = {r['group'] for r in inventory if r['split'] == 'test'}
    if train_groups & test_groups:
        raise ValueError('Parent leakage')
    source_cases = []
    with torch.inference_mode():
        for i, row in enumerate(inventory):
            if row['split'] != 'test':
                continue
            _, target = load_record(row)  # Revalidates source and annotation SHA.
            target = cv2.resize(target.astype(np.float32), (40,40), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            features = torch.from_numpy(np.array(cache[i],dtype=np.float32))[None]
            case = {'source': row['source'], 'view':row.get('view','original'), 'group':row['group'],
                    'image_sha256':row['image_sha256']}
            for name, model in models.items():
                prediction = model.model(features)[0].argmax(0).numpy()
                case[name] = metrics(prediction, target)
            source_cases.append(case)
    reference_report = json.loads(args.axis_reference.read_text())
    axis_cases = []
    dataset = args.dataset.resolve()
    with torch.inference_mode():
        for i, reference in enumerate(reference_report['cases']):
            path = (dataset / reference['source_path']).resolve()
            if not path.is_relative_to(dataset) or digest(path) != reference['source_sha256']:
                raise ValueError('Source identity mismatch')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                image = read_dxa(path)
            if image.pixel_sha256 != reference['pixel_sha256'] or image.image_uid != reference['image_uid']:
                raise ValueError('Decoded image identity mismatch')
            sx,sy = image.pixel_mm_x,image.pixel_mm
            reliable = np.isfinite([sx,sy]).all() and min(sx,sy)>0 and reference['scale_source'] not in ('device_default','',None)
            features = torch.from_numpy(spatial_features(image.pixels).astype(np.float32))[None]
            case = {k:reference[k] for k in ('source_path','study_group','pixel_sha256','axis_violation')}
            for name, model in models.items():
                logits = model.model(features)[0].numpy()
                regions = mask_regions(decode_masks(logits,image.pixels.shape))
                angle = numbered_axis(regions,sx,sy) if reliable else None
                case[name+'_angle_deg'] = angle
                case[name] = abs(angle)>5 if angle is not None else None
            axis_cases.append(case)
            if (i+1)%25 == 0:
                print(f'Compared organizer images {i+1}/{len(reference_report["cases"])}',flush=True)
    summary = {}
    for source,view in sorted({(c['source'],c['view']) for c in source_cases}):
        selected = [c for c in source_cases if (c['source'],c['view']) == (source,view)]
        summary[source+'/'+view] = {'images':len(selected)}
        for name in models:
            # Average within image first: an image with more named bodies has no extra weight.
            summary[source+'/'+view][name+'_mean_image_dice'] = float(np.mean([
                np.mean([v['dice'] for v in c[name]]) for c in selected]))
    report = {'protocol':'Paired frozen heads, identical cached features and holdout views; organizer fixed 5-degree axis threshold.',
              'source_summary':summary,'axis_summary':{name:score(axis_cases,name) for name in models},
              'source_cases':source_cases,'axis_cases':axis_cases,
              'model_sha256':{name:model.sha256 for name,model in models.items()},
              'training_report_sha256':digest(args.training/'evaluation.json'),
              'axis_reference_sha256':digest(args.axis_reference),'code_sha256':digest(Path(__file__)),
              'clinical_validation':False,'release_modified':False,'requirements_complete':False,
              'limitations':['Exploratory followup: source holdout and organizer errors were previously inspected.',
                             'AASCE groups are image identities, not established patient identities.',
                             'Hologic holdout contains only two patients and vendor ROI targets.',
                             'Binary axis labels do not validate vertebral numbering or partial Th12 coverage.',
                             'Derived crops use reference masks during training view preparation only; inference has no reference prompt.']}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('source_summary','axis_summary')},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('training','deployed','axis-reference','dataset','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
