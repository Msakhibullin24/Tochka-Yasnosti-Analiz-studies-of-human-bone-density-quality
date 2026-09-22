"""Train YOLO26x only from reviewed, study-disjoint DXA bbox/polygon annotations.

Manifest JSON: {task: detect|segment|pose, modality: DXA, names: [...], samples:
[{image: relative_path, label: relative_yolo_txt, study: group_id,
  split: train|val|test, reviewed: true}]}. Paths are relative to --dataset.
An explicitly reviewed empty label file means no target objects, not missing annotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml

from dxaqc.dicom_io import read_any
from dxaqc.specialist_qc import digest


def annotation(text, task, classes, keypoints=0):
    objects = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = np.array([float(x) for x in line.split()])
        if not np.isfinite(row).all() or len(row) < 5:
            raise ValueError('Invalid YOLO annotation')
        if row[0] != int(row[0]) or not 0 <= row[0] < classes:
            raise ValueError('Invalid class index')
        coordinates = row[1:] if task != 'pose' else np.concatenate([row[1:5], row[5:].reshape(-1,3)[:,:2].flatten()])
        if np.any((coordinates < 0) | (coordinates > 1)):
            raise ValueError('Coordinates must be normalized to [0, 1]')
        if task in ('detect','pose'):
            if len(row) != 5 + (3*keypoints if task == 'pose' else 0) or min(row[3:5]) <= 0:
                raise ValueError('Detection requires class xc yc width height')
            if np.any(row[1:3]-row[3:5]/2 < -1e-6) or np.any(row[1:3]+row[3:5]/2 > 1+1e-6):
                raise ValueError('Box extends outside image')
            if task == 'pose':
                visible = row[5:].reshape(-1,3)[:,2]
                if not np.isin(visible,[0,1,2]).all() or not np.any(visible > 0):
                    raise ValueError('Pose requires valid visibility and at least one labelled landmark')
        else:
            if len(row) < 7 or len(row) % 2 != 1:
                raise ValueError('Segmentation requires at least three polygon vertices')
            polygon = row[1:].reshape(-1,2).astype(np.float32)
            if cv2.contourArea(polygon) <= 0:
                raise ValueError('Polygon has zero area')
        objects.append(int(row[0]))
    return objects


def inspect_manifest(manifest, root):
    spec = json.loads(manifest.read_text())
    task, names = spec.get('task'), spec.get('names')
    if task not in ('detect', 'segment', 'pose') or spec.get('modality') != 'DXA':
        raise ValueError('Expected DXA detect/segment/pose manifest')
    if (not isinstance(names, list) or not names or any(not isinstance(n,str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('Unique, non-empty class names required')
    keypoint_names = spec.get('keypoint_names',[])
    if task == 'pose' and (not isinstance(keypoint_names,list) or not keypoint_names
                           or len(set(keypoint_names)) != len(keypoint_names)
                           or any(not isinstance(n,str) or not n.strip() for n in keypoint_names)):
        raise ValueError('Pose needs unique keypoint_names')
    samples = spec.get('samples')
    if not isinstance(samples, list) or not samples:
        raise ValueError('No annotated samples; image-level QC labels cannot train a detector')
    counts = {s: {'images': 0, 'negative_images': 0, 'objects': [0]*len(names)} for s in ('train','val','test')}
    root = root.resolve()
    studies, pixels, prepared = {}, set(), []
    for sample in samples:
        split, study = sample.get('split'), sample.get('study')
        if split not in counts or not isinstance(study,str) or not study.strip() or sample.get('reviewed') is not True:
            raise ValueError('Each sample needs a valid split, study group and reviewed=true')
        if studies.setdefault(study, split) != split:
            raise ValueError('Study leakage between partitions')
        paths = []
        for key in ('image','label'):
            if not isinstance(sample.get(key),str) or Path(sample[key]).is_absolute():
                raise ValueError('Image/label paths must be relative to dataset root')
            path = (root/sample[key]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Missing file or path outside dataset root')
            paths.append(path)
        image = read_any(paths[0])
        pixel_hash = hashlib.sha256(str(image.pixels.shape).encode()+image.pixels.tobytes()).hexdigest()
        if pixel_hash in pixels:
            raise ValueError('Duplicate decoded image; deduplicate before splitting')
        pixels.add(pixel_hash)
        label_text = paths[1].read_text()
        classes = annotation(label_text, task, len(names), len(keypoint_names))
        counts[split]['images'] += 1
        counts[split]['negative_images'] += int(not classes)
        for cls in classes: counts[split]['objects'][cls] += 1
        prepared.append((split, image.pixels, label_text, pixel_hash))
    if any(not c['images'] for c in counts.values()):
        raise ValueError('Train, val and test partitions must be non-empty')
    if any(n == 0 for n in counts['train']['objects']):
        raise ValueError('Each class needs training objects')
    if any(sum(counts[s]['objects']) == 0 for s in ('val','test')):
        raise ValueError('Validation and test require annotated positive objects')
    audit = {'task': task, 'names': names, 'counts': counts, 'manifest_sha256': digest(manifest), 'keypoint_names':keypoint_names,
             'study_counts': {s: sum(v == s for v in studies.values()) for s in counts},
             'scope': 'study-disjoint internal evaluation; patient identity not verified',
             'annotation_content_sha256': hashlib.sha256(''.join(p[3]+p[2] for p in prepared).encode()).hexdigest()}
    return audit, prepared


def json_value(value):
    """Normalize NumPy metric scalars and undefined values at the reporting boundary."""
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(args):
    if args.output.exists():
        raise ValueError('Choose a new output directory; do not overwrite experiments')
    if args.epochs < 1 or args.batch < 1 or args.threads < 1 or args.size < 64 or args.size % 32:
        raise ValueError('Invalid training settings; size must be a multiple of 32')
    audit, prepared = inspect_manifest(args.manifest, args.dataset)
    catalog = json.loads((Path(__file__).resolve().parents[1]/'docs/competition/specialist_sources.json').read_text())
    artifact_id = {'detect':'yolo26x_weights','segment':'yolo26x_seg_weights','pose':'yolo26x_pose_weights'}[audit['task']]
    spec = next(a for a in catalog['artifacts'] if a['id'] == artifact_id)
    weights = args.assets/spec['path']
    if digest(weights) != spec['sha256']:
        raise ValueError('Pretrained checkpoint checksum mismatch')
    args.output.mkdir(parents=True)
    (args.output/'data-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    dataset = args.output/'dataset'
    for i, (split, pixels, text, _) in enumerate(prepared):
        for folder in ('images','labels'): (dataset/folder/split).mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(dataset/'images'/split/f'{i:06}.png'), pixels):
            raise ValueError('Could not write training image')
        (dataset/'labels'/split/f'{i:06}.txt').write_text(text)
    del prepared
    config = {'path':str(dataset.resolve()), 'train':'images/train', 'val':'images/val',
              'test':'images/test', 'names': audit['names']}
    if audit['task'] == 'pose':
        config['kpt_shape'] = [len(audit['keypoint_names']),3]
        config['flip_idx'] = list(range(len(audit['keypoint_names'])))
    config_path = dataset/'dataset.yaml'
    config_path.write_text(yaml.safe_dump(config,allow_unicode=True))
    os.environ.setdefault('YOLO_CONFIG_DIR', str((args.assets/'ultralytics-config').resolve()))
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    status = {'status':'training','clinical_validation':False,'affects_decision':False}
    def save_status():
        (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save_status()
    try:
        model = YOLO(str(weights.resolve()))
        model.train(data=str(config_path.resolve()), epochs=args.epochs, imgsz=args.size,
                    batch=args.batch, device=args.device, workers=0, seed=17, deterministic=True,
                    optimizer='AdamW', lr0=.001, amp=False, plots=True, save=True,
                    project=str(args.output.resolve()), name='training', exist_ok=False,
                    hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0, scale=0, shear=0,
                    perspective=0, flipud=0, fliplr=0, mosaic=0, mixup=0, copy_paste=0)
        best = Path(model.trainer.best)
        if not best.is_file(): raise ValueError('Training produced no best checkpoint')
        checkpoint = args.output/'best.pt'
        shutil.copy2(best, checkpoint)
        candidate = YOLO(str(checkpoint))
        if any(not torch.isfinite(t).all() for t in candidate.model.state_dict().values()):
            raise ValueError('Training produced non-finite checkpoint parameters')
        metrics = {}
        for split in ('val','test'):
            result = candidate.val(data=str(config_path.resolve()), split=split, imgsz=args.size,
                                   batch=args.batch, device=args.device, workers=0, plots=True,
                                   project=str(args.output.resolve()), name='evaluation-'+split)
            metrics[split] = {'aggregate':{k:float(v) for k,v in result.results_dict.items()},
                              'per_class': result.summary()}
        report = {**audit,'metrics':metrics,'checkpoint_sha256':digest(checkpoint),
                  'pretrained_sha256':spec['sha256'],'clinical_validation':False,
                  'selection':'best checkpoint selected on val; test evaluated after training'}
        (args.output/'metrics.json').write_text(json.dumps(json_value(report),indent=2,allow_nan=False)+'\n')
        status['status'] = 'completed'
        save_status()
        return report
    except Exception as exc:
        status.update(status='failed',error=str(exc)); save_status(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=Path('data/specialists'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(inspect_manifest(args.manifest,args.dataset)[0],indent=2))
    else:
        run(args)
