"""Train and audit an isolated AP/lateral spine DXA candidate on public data.

Ramathibodi BMD spine exports and VFA exports supply the source view labels.
Conservative patient groups join matching folder IDs across both branches.
Organizer images are transfer checks only, never used to fit this model.
Nothing here changes the released QC pipeline or asserts clinical validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dxaqc.dicom_io import read_dxa
from dxaqc.embedding import embed, WEIGHTS_SHA256
from evaluate_organizer_dataset import binary_metrics, sha256

SOURCE_URL = 'https://data.mait.ai.in.th/dataset/bone-mineral-density-and-vertebral-fracture-asseessment-public-sharing'


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=5000,
                                                               random_state=17))


def variants(pixels):
    # Labels belong to views, not to native aspect ratio or bright-bone polarity.
    square = cv2.resize(pixels, (320, 320), interpolation=cv2.INTER_AREA)
    return (square, 255-square)


def features(pixels):
    return np.stack([embed(image) for image in variants(pixels)])


def source_patient_group(relative: Path) -> str:
    """Conservatively join numbered BMD/VFA folders despite zero padding."""
    if (len(relative.parts) != 5 or relative.parts[0] not in ('BMD', 'VFA')
            or relative.parts[1] not in ('Annotation', 'Verification')
            or not relative.parts[2].isascii() or not relative.parts[2].isdecimal()):
        raise ValueError('Unexpected public dataset layout')
    return str(int(relative.parts[2]))


def assessment(score: float) -> dict:
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid projection score')
    # This threshold is fixed, not selected against organizer mistakes. A score
    # from logistic regression is not a calibrated probability of correctness.
    view = 'lateral' if score >= .9 else 'frontal' if score <= .1 else 'unknown'
    return {'value': view, 'status': 'candidate' if view != 'unknown' else 'undetermined',
            'lateral_score': float(score), 'method': 'public_DXA_frozen_resnet18_lr',
            'clinical_validation': False, 'model_affects_decision': False}


def run(root: Path, dataset: Path, labels_path: Path, output: Path):
    import torch
    torch.set_num_threads(4)
    if output.exists():
        raise ValueError('Choose a new projection experiment directory')
    paths = sorted((root/'BMD').rglob('spine_image.dcm')) + sorted((root/'VFA').rglob('*.dcm'))
    if not paths or any(not (root/branch).is_dir() for branch in ('BMD', 'VFA')):
        raise ValueError('Both BMD and VFA source data are required')
    vectors, targets, groups, records = [], [], [], []
    pixel_groups, parents = {}, {}
    decoded = []
    duplicate_copies = 0
    def representative(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            group = parents[group]
        return group
    for path in paths:
        relative = path.relative_to(root)
        group = source_patient_group(relative)
        image = read_dxa(path)
        lateral = int(relative.parts[0] == 'VFA')
        representative(group)
        if image.pixel_sha256 in pixel_groups:
            old_group, old_view = pixel_groups[image.pixel_sha256]
            if old_view != lateral:
                raise ValueError('Identical pixels have contradictory projection labels')
            parents[representative(group)] = representative(old_group)
            duplicate_copies += 1
            continue
        pixel_groups[image.pixel_sha256] = (group, lateral)
        decoded.append((path, image, group, lateral))
    for path, image, source_group, lateral in decoded:
        group = representative(source_group)
        vectors.extend(features(image.pixels)); targets.extend([lateral]*2); groups.extend([group]*2)
        records.append({'source_sha256': sha256(path), 'pixel_sha256': image.pixel_sha256,
                        'patient_group': hashlib.sha256(group.encode()).hexdigest(),
                        'view': 'lateral' if lateral else 'frontal',
                        'label_basis': 'public_source_VFA_or_BMD_spine_branch'})
    X, y, g = np.asarray(vectors), np.asarray(targets), np.asarray(groups)
    if any(len(set(g[y == view])) < 5 for view in (0, 1)):
        raise ValueError('Five independent source groups per view are required')
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=17)
    probability = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)
    for fold, (train, test) in enumerate(folds.split(X, y, g)):
        assert not set(g[train]) & set(g[test])
        model = classifier().fit(X[train], y[train])
        probability[test] = model.predict_proba(X[test])[:, 1]
        fold_ids[test] = fold
    # Average the two deterministic representations of one image, not two test cases.
    scores = probability.reshape(-1, 2).mean(axis=1)
    actual = y.reshape(-1, 2)[:, 0]
    if not np.isfinite(scores).all():
        raise ValueError('Incomplete held-out projection predictions')
    model = classifier().fit(X, y)
    transfer, transfer_vectors, transfer_groups = [], [], []
    with labels_path.open(newline='') as stream:
        labels = [row for row in csv.DictReader(stream)
                  if row['region'] == 'spine' and row['quality_class'] in ('0', '1')]
    if not labels or any(not row.get('study_key') for row in labels):
        raise ValueError('Organizer spine transfer cohort requires study identities')
    for row in labels:
        path = (dataset/row['first_source_path']).resolve()
        if not path.is_relative_to(dataset.resolve()):
            raise ValueError('Organizer path escapes dataset')
        image = read_dxa(path)
        if image.pixel_sha256 in pixel_groups:
            raise ValueError('Source image overlaps the organizer transfer cohort')
        image_features = features(image.pixels)
        transfer_vectors.extend(image_features)
        transfer_groups.extend(['organizer:'+row['study_key']]*2)
        score = float(model.predict_proba(image_features)[:, 1].mean())
        transfer.append({'pixel_sha256': image.pixel_sha256, **assessment(score),
                         'forced_binary_view': 'lateral' if score >= .5 else 'frontal',
                         'reference_basis': 'organizer_AP_spine_cohort; not independently reviewed'})
    for index, record in enumerate(records):
        record.update(fold=int(fold_ids[index*2]), lateral_score=float(scores[index]))
    report = {'scope': 'isolated research candidate; released QC decisions unchanged',
              'evaluation_code_sha256': sha256(Path(__file__)),
              'source_url': SOURCE_URL, 'encoder_sha256': WEIGHTS_SHA256,
              'labels_sha256': sha256(labels_path), 'source_images': len(records),
              'duplicate_source_copies_collapsed': duplicate_copies,
              'source_patient_groups': len(set(groups)),
              'preprocessing': 'resize to square; average original and inverted polarity embeddings',
              'threshold': .5, 'threshold_basis': 'fixed before evaluation; no test-set optimization',
              'external_study_held_out': binary_metrics(actual, (scores >= .5).astype(int), scores),
              'organizer_AP_transfer': {'images': len(transfer),
                                        'forced_binary_predicted_lateral': sum(r['forced_binary_view']=='lateral' for r in transfer),
                                        'candidate_frontal': sum(r['value']=='frontal' for r in transfer),
                                        'candidate_lateral': sum(r['value']=='lateral' for r in transfer),
                                        'undetermined': sum(r['value']=='unknown' for r in transfer)},
              'abstention_thresholds': {'frontal_at_most': .1, 'lateral_at_least': .9,
                                       'basis': 'fixed research thresholds; score is not calibrated confidence'},
              'clinical_validation': False, 'model_affects_decision': False,
              'limitations': ['Protocol branch labels are not independent physician view annotations.',
                              'Matching numeric folder indices are conservatively grouped across branches; they are a proxy, not verified patient identities.',
                              'VFA and BMD differ in field of view, image texture and burned-in graphics.',
                              'No organizer-vendor lateral or oblique test images are available.',
                              'Binary classifier cannot recognize every unsupported projection.',
                              'Transfer count on AP-only images cannot establish lateral sensitivity.'],
              'source_oof': records, 'organizer_transfer': transfer}
    # A second, explicitly weak-label experiment learns the organizer AP domain.
    # Those AP labels come from the supplied cohort's protocol, not from a new
    # independent doctor assessment. All test studies stay outside training.
    adapted_X = np.vstack([X, np.asarray(transfer_vectors)])
    adapted_y = np.r_[y, np.zeros(len(transfer_vectors), dtype=int)]
    adapted_groups = np.r_[g, np.asarray(transfer_groups)]
    adapted_scores = np.full(len(adapted_y), np.nan)
    adapted_fold = np.full(len(adapted_y), -1)
    for fold, (train, test) in enumerate(folds.split(adapted_X, adapted_y, adapted_groups)):
        assert not set(adapted_groups[train]) & set(adapted_groups[test])
        fitted = classifier().fit(adapted_X[train], adapted_y[train])
        adapted_scores[test] = fitted.predict_proba(adapted_X[test])[:, 1]
        adapted_fold[test] = fold
    adapted_scores = adapted_scores.reshape(-1, 2).mean(axis=1)
    adapted_actual = adapted_y.reshape(-1, 2)[:, 0]
    adapted_assessments = [assessment(float(s)) for s in adapted_scores]
    report['AP_domain_adaptation'] = {
        'protocol': 'five group-held-out folds; source branch labels plus organizer weak AP protocol labels',
        'clinical_validation': False,
        'all_images': binary_metrics(adapted_actual, (adapted_scores >= .5).astype(int), adapted_scores),
        'organizer_AP_predicted_lateral': int((adapted_scores[len(records):] >= .5).sum()),
        'organizer_AP_undetermined': sum(r['value']=='unknown' for r in adapted_assessments[len(records):]),
        'held_out_scores': [{'pixel_sha256': record['pixel_sha256'],
                             'fold': int(adapted_fold[index*2]), **adapted_assessments[index]}
                            for index, record in enumerate(records+transfer)],
        'limitations': ['Organizer AP protocol labels are weak references, not independent view annotations.',
                        'No GE lateral sensitivity can be measured from this cohort.']}
    output.mkdir(parents=True)
    joblib.dump({'schema_version': 1, 'status': 'research_only', 'encoder_sha256': WEIGHTS_SHA256,
                 'model': model, 'threshold': .5, 'classes': ['frontal', 'lateral'],
                 'clinical_validation': False}, output/'projection.joblib')
    report['model_sha256'] = sha256(output/'projection.joblib')
    adapted_model = classifier().fit(adapted_X, adapted_y)
    joblib.dump({'schema_version': 1, 'status': 'research_only', 'encoder_sha256': WEIGHTS_SHA256,
                 'model': adapted_model, 'threshold': .5, 'classes': ['frontal', 'lateral'],
                 'reference_basis': 'public branch labels plus organizer weak AP protocol labels',
                 'clinical_validation': False}, output/'projection_adapted.joblib')
    report['AP_domain_adaptation']['model_sha256'] = sha256(output/'projection_adapted.joblib')
    (output/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def predict(model_path: Path, input_path: Path) -> dict:
    """Run only a locally trained candidate, explicitly outside released QC."""
    import torch
    torch.set_num_threads(4)
    bundle = joblib.load(model_path)
    if (bundle.get('schema_version') != 1 or bundle.get('status') != 'research_only'
            or bundle.get('encoder_sha256') != WEIGHTS_SHA256
            or bundle.get('classes') != ['frontal', 'lateral']):
        raise ValueError('Incompatible projection candidate')
    model = bundle['model']
    if not np.array_equal(model.classes_, np.array([0, 1])):
        raise ValueError('Projection model class order differs from its contract')
    image = read_dxa(input_path)
    result = assessment(float(model.predict_proba(features(image.pixels))[:, 1].mean()))
    result['method'] = bundle.get('method', result['method'])
    return {**result, 'model_sha256': sha256(model_path), 'pixel_sha256': image.pixel_sha256,
            'scope': 'spine research candidate only; no clinical QC verdict'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Run an existing local research candidate')
    parser.add_argument('--input', type=Path, help='Spine DICOM for candidate inference')
    for name in ('root', 'dataset', 'labels', 'output'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*', category=UserWarning)
        if args.model:
            if not args.input or any((args.root, args.dataset, args.labels, args.output)):
                parser.error('Inference requires --model and --input only')
            print(json.dumps(predict(args.model, args.input)))
        else:
            if args.input or not all((args.root, args.dataset, args.labels, args.output)):
                parser.error('Training requires --root, --dataset, --labels and --output')
            report = run(args.root, args.dataset, args.labels, args.output)
            print(json.dumps({key: report[key] for key in ('external_study_held_out', 'organizer_AP_transfer')}))
            print(json.dumps({key: value for key, value in report['AP_domain_adaptation'].items()
                              if key not in ('held_out_scores', 'limitations')}))
