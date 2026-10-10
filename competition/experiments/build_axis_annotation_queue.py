"""P0: build the expert annotation queue for the spine_axis definition.

The closure plan makes this the first blocking item, and it is the one thing in the
whole programme that no amount of modelling can substitute for. spine_axis carries 10
positives and an F1 of 0.167; the measured angle separates the classes at AUC 0.834,
but the official 5 degree rule fires on 1 of 10. An independently trained regressor on
synthetic rotations found 0 of 10 above 5 degrees and predicted the labelled positives
to be *less* tilted than the aligned frames. So either the measurement is biased toward
zero or the label does not denote an in-plane rotation, and only a specialist can say
which.

This produces the queue, not the answers. The plan is explicit that the absence of
expert replies must never be treated as a label source, so every expert field is empty
and the package carries no ground truth. It contains:

  * all 10 labelled spine_axis positives;
  * every row where the measurer and the classifier disagree, which is where a
    definition error would show up;
  * a seeded sample of aligned negatives to anchor the scale.

Measured angle, the classifier score and the official threshold are attached read-only so
the specialist can see what the pipeline currently believes, but the verdict fields are
free text and numeric and start empty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gpu_research.common import dump  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402

AXIS_LIMIT_DEG = 5.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--classifier', type=Path, required=True,
                        help='An OOF CSV whose criterion_scores contain spine_axis')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--negatives', type=int, default=40)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)

    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    baseline = pd.read_csv(args.baseline).sort_values('index').reset_index(drop=True)
    classifier = pd.read_csv(args.classifier).sort_values('index').reset_index(drop=True)
    if not np.array_equal(baseline['index'].to_numpy(), classifier['index'].to_numpy()):
        raise ValueError('Baseline and classifier rows are not aligned')
    audit = inspect_sources([('organiser', args.labels, args.dataset)])
    audited = {r['path']: r['pixel_sha256_current'] for r in audit['entries']}

    classifier_scores, classifier_thresholds = [], []
    for column in ('criterion_scores', 'criterion_thresholds'):
        parsed = []
        for v in classifier[column]:
            parsed.append(json.loads(v) if isinstance(v, str) and v else {})
        if column == 'criterion_scores':
            classifier_scores = parsed
        else:
            classifier_thresholds = parsed

    spine = labels.region.eq('spine').to_numpy()
    label_axis = pd.to_numeric(labels.spine_axis, errors='coerce').to_numpy(float)
    records = {}
    for i in range(len(baseline)):
        if not spine[i]:
            continue
        row = baseline.iloc[i]
        states = json.loads(row['criterion_states'])
        angle = states.get('spine_axis', {}).get('angle_deg')
        score = classifier_scores[i].get('spine_axis')
        measurer_fails = bool(angle is not None and np.isfinite(angle) and abs(angle) > AXIS_LIMIT_DEG)
        # The classifier's own recorded threshold, not 0.5. Its scores are within-split
        # ranks, so 0.5 would flag roughly half the cohort and manufacture fake
        # disagreements: 44 of 99 spine rows were reported as disagreements that way.
        recorded = classifier_thresholds[i].get('spine_axis')
        classifier_fails = bool(score is not None and recorded is not None
                                and np.isfinite(recorded) and score >= recorded)
        entry = {
            'index': int(row['index']),
            'first_source_path': labels['first_source_path'].iloc[i],
            'study': row['study'],
            'image_sha256': audited.get(labels['first_source_path'].iloc[i]),
            'rows': int(labels['rows'].iloc[i]), 'columns': int(labels['columns'].iloc[i]),
            'original_spine_axis_label': None if not np.isfinite(label_axis[i]) else int(label_axis[i]),
            'measured_angle_deg': None if angle is None else float(angle),
            'official_limit_deg': AXIS_LIMIT_DEG,
            'measurer_says_fail': measurer_fails,
            'classifier_score': None if score is None else float(score),
            'classifier_threshold': None if recorded is None else float(recorded),
            'classifier_says_fail': classifier_fails,
            'selection_reason': None,
            # Expert fields. All empty on purpose. Never auto-filled.
            'expert_axis_is_measurable_here': None,
            'expert_landsmarks_usable': None,
            'expert_segment_used': None,
            'expert_angle_deg': None,
            'expert_defect_present': None,
            'expert_defect_kind': None,
            'expert_defect_kind_other': None,
            'expert_label_agrees_with_original': None,
            'expert_change_reason': None,
            'expert_notes': None,
        }
        if entry['original_spine_axis_label'] == 1:
            entry['selection_reason'] = 'labelled positive'
        elif measurer_fails != classifier_fails:
            entry['selection_reason'] = 'measurer and classifier disagree'
        elif entry['original_spine_axis_label'] == 0 and measurer_fails and abs(float(angle)) > 3.0:
            entry['selection_reason'] = 'aligned label but measured angle near the limit'
        # Keep every spine row in the pool. The plan asks for 30-50 aligned negatives as
        # an anchor for the specialist's scale, and those rows carry no selection reason of
        # their own - they are the ones where the pipeline already agrees with the label.
        entry['selection_reason'] = entry['selection_reason'] or 'aligned negative, no disagreement'
        records[i] = entry

    priority = [records[i] for i in sorted(records)
                if records[i]['selection_reason'] in ('labelled positive', 'measurer and classifier disagree')]
    rng = np.random.default_rng(args.seed)
    pool = sorted(i for i in records
                  if records[i]['selection_reason'] == 'aligned negative, no disagreement')
    chosen = rng.permutation(pool)[:args.negatives]
    sample = [records[int(i)] for i in chosen]
    queue = priority + sample

    fields = ['index', 'first_source_path', 'study', 'image_sha256', 'rows', 'columns',
              'original_spine_axis_label', 'measured_angle_deg', 'official_limit_deg',
              'measurer_says_fail', 'classifier_score', 'classifier_threshold',
              'classifier_says_fail', 'selection_reason']
    expert_fields = ['expert_axis_is_measurable_here', 'expert_landsmarks_usable', 'expert_segment_used',
                     'expert_angle_deg', 'expert_defect_present', 'expert_defect_kind',
                     'expert_defect_kind_other', 'expert_label_agrees_with_original',
                     'expert_change_reason', 'expert_notes']
    frame = pd.DataFrame(queue)[fields + expert_fields]
    frame.to_csv(args.output / 'spine_axis_annotation_queue.csv', index=False)

    template = {'criteria_version': 'v1-unverified', 'axis_limit_deg': AXIS_LIMIT_DEG,
                'instructions_ru': [
                    'Это очередь разметки, а не набор ответов. Все поля expert_* пусты.',
                    'Задача: определить, означает ли метка spine_axis наклон позвоночника в плоскости кадра, '
                    'либо смещение, сколиоз или иной дефект укладки.',
                    'Заполнить expert_axis_is_measurable_here: можно ли вообще измерить ось в этом кадре.',
                    'Заполнить expert_landmarks_usable: пригодны ли верхний и нижний ориентир для оценки оси.',
                    'Заполнить expert_angle_deg: численный угол в градусах, измеренный в физической сетке с '
                    'раздельным масштабом X и Y.',
                    'Заполнить expert_defect_present и expert_defect_kind: наклон, смещение, сколиоз, другое.',
                    'Заполнить expert_label_agrees_with_original и expert_change_reason при расхождении.',
                    'Не заполнять expert_* по умолчанию. Отсутствие ответа означает unknown, а не отрицательную метку.'],
                'original_labels_untouched': True}
    template['queue_file'] = 'spine_axis_annotation_queue.csv'
    template['rows_jsonl'] = 'annotation_queue.jsonl'
    (args.output / 'README.json').write_text(json.dumps(template, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'annotation_queue.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in queue) + '\n')

    dump(args.output / 'queue_manifest.json', {
        'protocol': __doc__,
        'rows_total': len(queue),
        'rows_labelled_positive': sum(1 for r in queue if r['selection_reason'] == 'labelled positive'),
        'rows_measurer_classifier_disagree': sum(1 for r in queue
                                                 if r['selection_reason'] == 'measurer and classifier disagree'),
        'rows_sampled_negatives': len(sample),
        'rows_aligned_negative_anchor_pool': int(
            sum(1 for r in records.values() if r['selection_reason'] == 'aligned negative, no disagreement')),
        'labels_sha256': hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        'baseline_sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        'classifier_oof_sha256': hashlib.sha256(args.classifier.read_bytes()).hexdigest(),
        'source_fingerprint': hashlib.sha256(json.dumps(
            [(r['path'], r['pixel_sha256_current']) for r in audit['entries']],
            ensure_ascii=False).encode()).hexdigest(),
        'contains_ground_truth': False,
        'warning': ('The package deliberately contains no expert answers. Missing replies are unknown, never negative '
                    'labels. Until a specialist fills it, this project has no verified reference for the spine_axis '
                    'definition, and R04 stays open.'),
        'clinical_validation': False, 'model_affects_decision': False})

    print(f'queue rows: {len(queue)}')
    counts = pd.Series([r['selection_reason'] for r in queue]).value_counts().to_dict()
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print('\nexample row (expert fields empty):')
    print(json.dumps(queue[0], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()