"""Add whole-spine synthetic AP views; keep real-data patient folds unchanged.

AASCE transfer is diagnostic: synthetic images may derive from that source.
Fresh CDC lateral images are transfer data only, never training labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

from dxaqc.embedding import WEIGHTS_SHA256
from dxaqc.dicom_io import read_dxa
from evaluate_organizer_dataset import sha256
from evaluate_projection_candidate import features, classifier, assessment
from train_multisource_projection import source_weights, evaluate


def run(base, synthetic, aasce, nhanes, ukb, output):
    if output.exists():
        raise ValueError('Choose a new experiment directory')
    torch.set_num_threads(4)
    original = json.loads((base/'evaluation.json').read_text())
    if sha256(base/'features.npz') != original['feature_sha256'] or original['encoder_sha256'] != WEIGHTS_SHA256:
        raise ValueError('Frozen feature source changed')
    with np.load(base/'features.npz', allow_pickle=False) as cache:
        real_x = cache['features']
    rows = original['cases'];real_y = np.array([r['target'] for r in rows])
    sources = np.array([r['source'] for r in rows]);folds = np.array([r['fold'] for r in rows])
    seen = {r['pixel_sha256'] for r in rows}
    paths = sorted(synthetic.glob('Spinal-AI2024-subset[1-4]/*.jpg'))
    if len(paths) != 16000:
        raise ValueError('Expected official 16000-image synthetic training split')
    selected = sorted(np.random.default_rng(17).choice(len(paths), 512, replace=False))
    vectors, synthetic_rows = [], []
    for i in selected:
        p = paths[i];pixels = cv2.imread(str(p), 0)
        if pixels is None:
            raise ValueError('Unreadable synthetic image')
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError('Duplicate training pixels')
        seen.add(digest)
        vectors.append(features(pixels));synthetic_rows.append({'file':str(p.relative_to(synthetic)),
            'source_sha256':sha256(p),'pixel_sha256':digest,'target':0,'reference_basis':'synthetic whole-spine AP source convention'})
        if len(vectors)%100 == 0:
            print(f'Embedded synthetic {len(vectors)}/512', flush=True)
    synthetic_x = np.array(vectors)

    def fit(mask):
        x = np.concatenate([real_x[mask], synthetic_x]);y = np.r_[real_y[mask], np.zeros(len(synthetic_x), dtype=int)]
        domain = np.r_[sources[mask], np.repeat('synthetic_whole_spine', len(synthetic_x))]
        return classifier().fit(x.reshape(-1, x.shape[-1]), np.repeat(y, 2),
                                 logisticregression__sample_weight=np.repeat(source_weights(domain), 2))

    scores = np.full(len(rows), np.nan)
    for fold in range(5):
        test = folds == fold
        if {r['group'] for r, t in zip(rows, test) if t} & {r['group'] for r, t in zip(rows, test) if not t}:
            raise ValueError('Real patient overlap')
        model = fit(~test)
        scores[test] = model.predict_proba(real_x[test].reshape(-1, real_x.shape[-1]))[:, 1].reshape(-1, 2).mean(axis=1)
    model = fit(np.ones(len(rows), dtype=bool))
    previous = joblib.load(base/'projection_multisource.joblib')['model']
    output.mkdir(parents=True)
    model_path = output/'projection_augmented.joblib'
    joblib.dump({'schema_version':1,'status':'research_only','encoder_sha256':WEIGHTS_SHA256,
                 'model':model,'threshold':.5,'classes':['frontal','lateral'],
                 'clinical_validation':False,'method':'whole_spine_augmented_frozen_resnet18_lr'}, model_path)
    transfer = []
    candidates = [(p, 0, 'aasce', 'source AP; potential synthetic parent overlap') for p in sorted(aasce.glob('*.jpg'))]
    candidates += [(p, 1, 'nhanes', 'CDC L-prefix lateral lumbar; independent source') for p in sorted(nhanes.glob('L*.tiff'))]
    for p in sorted(ukb.rglob('*.dcm')):
        if '20180719141305.1.2.12.1' in p.name:
            candidates.append((p, 0, 'ukb', 'AI visual assessment of official format example'))
        elif '20180719141308.1.7.12.1' in p.name:
            candidates.append((p, 1, 'ukb', 'AI visual assessment of official format example'))
    for path, target, domain, basis in candidates:
        if path.suffix == '.dcm':
            image = read_dxa(path);pixels = image.pixels;digest = image.pixel_sha256
        elif path.suffix == '.tiff':
            native = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if native is None or native.ndim != 2:
                raise ValueError('Invalid CDC raster')
            pixels = cv2.normalize(native, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        else:
            pixels = cv2.imread(str(path), 0);digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError('Exact training pixel overlap in transfer')
        vector = features(pixels)
        before, after = [float(m.predict_proba(vector)[:,1].mean()) for m in (previous, model)]
        transfer.append({'source':domain,'target':target,'reference_basis':basis,'source_sha256':sha256(path),
                         'pixel_sha256':digest,'baseline':assessment(before),'augmented':assessment(after)})
    report = {'clinical_validation':False,'release_modified':False,
              'protocol':'Fixed 512 synthetic training AP images, seed17; equal domain mass; real five patient folds preserved; fixed LR C=.01 and thresholds .5/.1/.9; no transfer tuning',
              'held_out_real_sources':evaluate(rows,scores),'baseline_real_sources':original['held_out_by_source'],
              'transfer_by_source': {name:evaluate([{'source':r['source'],'target':r['target']} for r in transfer],
                                           np.array([r[name]['lateral_score'] for r in transfer])) for name in ('baseline','augmented')},
              'transfer_cases':transfer,'synthetic_training_cases':synthetic_rows,
              'model_sha256':sha256(model_path),'code_sha256':sha256(Path(__file__)),
              'limitations':['Synthetic augmentation is not physician-labelled clinical data.',
                              'AASCE may be a source of synthetic parents and was previously examined; its score is diagnostic, not independent validation.',
                              'Source projection labels and two AI-read UKB examples do not validate clinical DXA QC.',
                              'CDC TIFF converted to uint8 by native min-max scaling; source-label performance only.']}
    (output/'evaluation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report['transfer_by_source'],indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','synthetic','aasce','nhanes','ukb','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();run(a.base,a.synthetic,a.aasce,a.nhanes,a.ukb,a.output)
