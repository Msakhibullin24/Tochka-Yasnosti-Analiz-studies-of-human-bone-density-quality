"""Train a source-only projection candidate; hold out BUU patients and all DXA.

Fixed model/threshold, no tuning against the held-out views or DXA transfer data.
The output is research-only and cannot attest vendor-specific lateral DXA QC.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from dxaqc.dicom_io import read_dxa
from dxaqc.embedding import WEIGHTS_SHA256
from evaluate_organizer_dataset import binary_metrics, sha256
from evaluate_projection_candidate import classifier, features, assessment


def patient_split(records):
    patients = sorted({r['patient_id'] for r in records})
    if len(patients) < 4:
        raise ValueError('At least four independent patients are required')
    rng = np.random.default_rng(17)
    held_out = set(rng.choice(patients, size=len(patients)//4, replace=False))
    test = np.array([r['patient_id'] in held_out for r in records])
    if ({r['pixel_sha256'] for r, t in zip(records, test) if t}
            & {r['pixel_sha256'] for r, t in zip(records, test) if not t}):
        raise ValueError('Pixel leakage across held-out patients')
    return patients, held_out, test


def run(root, ramathibodi, organizer, labels, output):
    import torch
    torch.set_num_threads(4)
    if output.exists():
        raise ValueError('Choose a new experiment directory')
    reference = json.loads(Path('competition/reports/buu_frozen_transfer_2026_09_26.json').read_text())
    records = reference['cases']
    patients, held_out, test = patient_split(records)
    vectors = []
    for i, record in enumerate(records):
        path = root/record['file']
        if sha256(path) != record['source_sha256']:
            raise ValueError('BUU source changed since inventory')
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        vectors.append(features(image))
        if (i+1) % 100 == 0:
            print(f'Embedded {i+1}/{len(records)}', flush=True)
    X = np.asarray(vectors)
    y = np.array([r['lateral_reference'] for r in records])
    model = classifier().fit(X[~test].reshape(-1, X.shape[-1]), np.repeat(y[~test], 2))
    scores = model.predict_proba(X[test].reshape(-1, X.shape[-1]))[:, 1].reshape(-1, 2).mean(axis=1)
    transfer, seen = [], set()
    paths = [(p, int('VFA' in p.relative_to(ramathibodi).parts), 'public_DXA_branch')
             for p in sorted((ramathibodi/'BMD').rglob('spine_image.dcm'))+sorted((ramathibodi/'VFA').rglob('*.dcm'))]
    with labels.open() as stream:
        paths += [(organizer/r['first_source_path'], 0, 'weak_organizer_AP_protocol')
                  for r in csv.DictReader(stream) if r['region']=='spine' and r['quality_class'] in ('0','1')]
    for path, target, basis in paths:
        image = read_dxa(path)
        if image.pixel_sha256 in seen:
            continue
        seen.add(image.pixel_sha256)
        score = float(model.predict_proba(features(image.pixels))[:, 1].mean())
        transfer.append({'pixel_sha256': image.pixel_sha256, 'lateral_reference': target,
                         'reference_basis': basis, **assessment(score),
                         'method':'BUU_only_frozen_resnet18_lr'})
    output.mkdir(parents=True)
    joblib.dump({'schema_version':1, 'status':'research_only', 'encoder_sha256':WEIGHTS_SHA256,
                 'model':model, 'threshold':.5, 'classes':['frontal','lateral'],
                 'clinical_validation':False, 'training_source':'BUU 300 patients only'}, output/'projection_buu.joblib')
    metrics = {}
    for basis in ('public_DXA_branch', 'weak_organizer_AP_protocol'):
        group = [r for r in transfer if r['reference_basis']==basis]
        actual = np.array([r['lateral_reference'] for r in group])
        probability = np.array([r['lateral_score'] for r in group])
        metrics[basis] = {'metrics':binary_metrics(actual,(probability>=.5).astype(int),probability),
                         'undetermined':sum(r['value']=='unknown' for r in group)}
    report = {'training_source':'BUU only; no DXA fitted', 'training_patients':len(patients)-len(held_out),
              'held_out_patients':len(held_out), 'held_out_images':int(test.sum()),
              'clinical_validation':False, 'model_affects_decision':False,
              'protocol':'Fixed C=.01 logistic classifier and .5 threshold; seeded patient split before fitting; original/inverted representations stay in the same split',
              'held_out_buu_metrics':binary_metrics(y[test],(scores>=.5).astype(int),scores),
              'held_out_buu_undetermined':sum(assessment(float(s))['value']=='unknown' for s in scores),
              'transfer':metrics, 'transfer_cases':transfer,
              'held_out_cases':[{**r,'lateral_score':float(s)} for r,s in zip([r for r,t in zip(records,test) if t],scores)],
              'model_sha256':sha256(output/'projection_buu.joblib'), 'code_sha256':sha256(Path(__file__)),
              'limitations':['Plain radiographs are not DXA.', 'No lateral GE DXA reference.',
                             'Organizer AP references are weak protocol labels.', 'Only AP/lateral spine, not unsupported views or hip projection.']}
    (output/'evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ('held_out_buu_metrics','held_out_buu_undetermined','transfer')},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','ramathibodi','organizer','labels','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    a=parser.parse_args()
    run(a.root,a.ramathibodi,a.organizer,a.labels,a.output)
