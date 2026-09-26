"""Nested study-fold training of coherent quality/type decisions with train-only artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from dxaqc.dicom_io import read_dxa
from dxaqc.embedding import embed
from dxaqc.geometry import measure_image
from dxaqc.joint_quality import JointQualityModel, select_quality_threshold
from dxaqc.learned_anatomy import LearnedAnatomy, numbered_axis
from dxaqc.model import best_f1_threshold, group_of, official_violation_type
from dxaqc.synthetic_defects import artifact_pair, rotate_expand
from evaluate_organizer_dataset import evaluate, sha256
from source_integrity import inspect_sources
from structured_typifier import target
from train import build_table, HERE


def run(args):
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    torch.set_num_threads(4)
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    fingerprint = hashlib.sha256(json.dumps([(x['path'], x['pixel_sha256_current']) for x in audit['entries']]).encode()).hexdigest()
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    features, embeddings, _ = build_table(args.dataset, labels, HERE/'.cache', fingerprint)
    # Never mutate the shared feature cache used by the baseline experiments.
    features = [dict(f) for f in features]
    axis_cases = []
    if args.anatomy_model:
        anatomy_model = LearnedAnatomy(args.anatomy_model)
        for i, label in enumerate(labels.itertuples()):
            if label.region != 'spine':
                continue
            image = read_dxa(args.dataset/label.first_source_path)
            prediction = anatomy_model.predict(image.pixels, 'spine')
            angle = numbered_axis(prediction['regions'], image.pixel_mm_x or image.pixel_mm, image.pixel_mm)
            axis_cases.append({'source_path': label.first_source_path, 'angle_deg': angle,
                               'regions': prediction['regions'], 'fallback': angle is None})
            if angle is not None:
                features[i].update(spine_abs_angle_deg=abs(angle), spine_angle_deg=angle)
        print(f'Learned-mask axes available {sum(x["angle_deg"] is not None for x in axis_cases)}/{len(axis_cases)}', flush=True)
    with args.labels.open(newline='') as stream:
        reference = {r['first_source_path']: r for r in csv.DictReader(stream) if r['quality_class'] in ('0', '1')}
    with args.baseline.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    evaluate(args.labels, args.baseline, repeats=20)
    index = {p: i for i, p in enumerate(labels.first_source_path)}
    targets = [target(reference[p]) for p in labels.first_source_path]
    groups = labels.study_key.to_numpy()
    anatomy = [group_of(r) for r in labels.region]
    auxiliary = []
    if args.artifacts or args.rotations:
        for i, label in enumerate(labels.itertuples()):
            if label.region != 'spine' or targets[i] != 'normal':
                continue
            image = read_dxa(args.dataset/label.first_source_path)
            h, w = image.pixels.shape
            variants = []
            if args.artifacts:
                altered, _, metadata = artifact_pair(image.pixels, {}, (w//3, h//3, w//3+max(2,w//30), h//3+max(5,h//12)))
                variants.append((altered, 'spine_artifact', metadata))
            if args.rotations and image.pixel_mm_source not in ('device_default', '', None):
                spacing = (image.pixel_mm_x or image.pixel_mm, image.pixel_mm)
                for degrees in (-15., 15.):
                    altered, _ = rotate_expand(image.pixels, degrees, spacing)
                    variants.append((altered, 'spine_axis', {'target_origin': 'expert_normal_plus_geometric_bound',
                                     'relative_degrees': degrees, 'original_abs_angle_upper_bound': 5.,
                                     'synthetic_abs_angle_lower_bound': 10., 'field_clipped': False}))
                control, _ = rotate_expand(altered, -15., spacing)
                variants.append((control, 'normal', {'target_origin': 'inverse_rotation_control', 'net_degrees': 0.}))
            for altered, label_target, metadata in variants:
                auxiliary.append({'parent_index': i, 'features': measure_image(altered, 'spine', image.pixel_mm, image.pixel_mm_x).features,
                                  'embedding': embed(altered), 'target': label_target, 'group': groups[i],
                                  'metadata': metadata, 'pixel_sha256': hashlib.sha256(altered.tobytes()).hexdigest()})
        print(f'Prepared {len(auxiliary)} train-only synthetic children', flush=True)

    def fit(indices, group):
        selected = [i for i in indices if anatomy[i] == group and targets[i] is not None]
        children = [a for a in auxiliary if a['parent_index'] in selected] if group == 'spine' else []
        fs = [features[i] for i in selected] + [a['features'] for a in children]
        em = np.array([embeddings[i] for i in selected] + [a['embedding'] for a in children])
        ys = [targets[i] for i in selected] + [a['target'] for a in children]
        return JointQualityModel(group, args.axis_policy).fit(fs, em, ys)

    def calibrate(indices, group):
        indices = np.array([i for i in indices if anatomy[i] == group])
        scores = np.full(len(indices), np.nan)
        for inner_train, inner_test in GroupKFold(3).split(indices, groups=groups[indices]):
            model = fit(indices[inner_train], group)
            scores[inner_test] = [d['score'] for d in model.distributions(
                [features[i] for i in indices[inner_test]], embeddings[indices[inner_test]])]
        known = np.array([targets[i] is not None for i in indices])
        if not np.isfinite(scores).all():
            raise ValueError('Incomplete inner OOF')
        forced = [args.axis_policy == 'measured' and group == 'spine' and float(features[i].get('spine_abs_angle_deg', 0) or 0) > 5. for i in indices]
        return select_quality_threshold(labels.quality_class.to_numpy()[indices][known].astype(int), scores[known], np.asarray(forced)[known])

    predictions = {}
    split_audit = []
    for fold in sorted({r['fold'] for r in rows}):
        held_studies = {r['study'] for r in rows if r['fold'] == fold}
        training = np.array([i for i, study in enumerate(groups) if study not in held_studies])
        for group in ('spine', 'hip'):
            model = fit(training, group)
            model.quality_threshold = calibrate(training, group)
            held = [r for r in rows if r['fold'] == fold and group_of(r['predicted_region']) == group]
            testing = [index[r['source_path']] for r in held]
            for row, decision in zip(held, model.decisions([features[i] for i in testing], embeddings[testing])):
                result = dict(row)
                result.update(quality_pred=str(decision['quality']), quality_score=str(decision['score']),
                              quality_threshold=str(decision['quality_threshold']),
                              violation_type=official_violation_type(decision['violations']),
                              decision_version=decision['decision_version'], decision_reason=decision['decision_reason'],
                              criterion_states=json.dumps(decision['criterion_states'], sort_keys=True),
                              criterion_scores='', joint_evidence=json.dumps(decision['joint_evidence'], sort_keys=True))
                predictions[row['source_path']] = result
            train_studies = set(groups[training])
            if held_studies & train_studies:
                raise ValueError('Outer study leakage')
            split_audit.append({'fold': fold, 'group': group, 'training_studies': sorted(train_studies),
                                'held_studies': sorted(held_studies), 'threshold': model.quality_threshold})
        print(f'Completed nested joint fold {fold}', flush=True)
    candidates = [predictions[r['source_path']] for r in rows]
    args.output.mkdir(parents=True)
    path = args.output/'joint_oof.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) + ['joint_evidence'])
        writer.writeheader(); writer.writerows(candidates)
    metrics = evaluate(args.labels, path, repeats=200)
    models = {}
    for group in ('spine', 'hip'):
        models[group] = fit(np.arange(len(labels)), group)
        models[group].quality_threshold = calibrate(np.arange(len(labels)), group)
    joblib.dump(models, args.output/'joint_models.joblib')
    report = {'protocol': 'Existing outer study folds; inner 3-fold thresholds; synthetic children follow parents at every split; fixed logistic C=.01',
              'metrics': metrics, 'fold_audit': split_audit, 'synthetic_children': len(auxiliary),
              'synthetic_targets': {t: sum(a['target'] == t for a in auxiliary) for t in sorted({a['target'] for a in auxiliary})},
              'axis_policy': args.axis_policy,
              'excluded_training_targets': sum(t is None for t in targets),
              'labels_sha256': sha256(args.labels), 'baseline_sha256': sha256(args.baseline),
              'model_sha256': sha256(args.output/'joint_models.joblib'), 'code_sha256': sha256(Path(__file__)),
              'joint_code_sha256': sha256(HERE/'dxaqc/joint_quality.py'), 'clinical_validation': False,
              'anatomy_model_sha256': sha256(args.anatomy_model) if args.anatomy_model else None,
              'anatomical_axis_cases': axis_cases,
              'limitations': ['Internal previously inspected studies, not an independent clinical test.',
                              'Axis feasibility uses current geometric measurements; their error remains a limitation.',
                              'A fixed final model on all studies is not its OOF instance.',
                              'Synthetic objects do not supply missing ROI or anatomical ground truth.']}
    (args.output/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'f1': metrics['overall_quality']['f1'], 'sensitivity': metrics['overall_quality']['sensitivity'],
                      'macro_type_f1': metrics['criterion_macro_f1'], 'untyped': metrics['predicted_positive_without_type']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'labels', 'baseline', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--artifacts', action='store_true')
    parser.add_argument('--rotations', action='store_true')
    parser.add_argument('--axis-policy', choices=['measured', 'advisory'], default='measured')
    parser.add_argument('--anatomy-model', type=Path)
    args = parser.parse_args(); run(args)
