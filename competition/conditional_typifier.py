"""Research-only type decoding from a learned distribution, preserving detection.

Conditioning on an existing alarm is a model hypothesis, not new ground truth.
The conditional score is neither calibrated confidence nor the alarm probability.
"""
from __future__ import annotations

import json
import math

from dxaqc.decision import AXIS_LIMIT_DEG
from dxaqc.model import CRITERIA, REGIONS, group_of, official_violation_type


def apply_conditional_type(row, prediction):
    result = dict(row)
    if row['quality_pred'] not in ('0', '1'):
        raise ValueError('Invalid binary prediction')
    if row['quality_pred'] == '0' or row['violation_type'].strip():
        return result
    if row['predicted_region'] not in REGIONS:
        raise ValueError('Unsupported anatomy')
    allowed = CRITERIA[group_of(row['predicted_region'])]
    scores = prediction['combination_scores']
    if 'normal' not in scores or not scores:
        raise ValueError('Need learned normal/type probabilities')
    for combination, score in scores.items():
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid type probability')
        codes = combination.split(';')
        if combination != 'normal' and (len(set(codes)) != len(codes) or not set(codes) <= allowed.keys()):
            raise ValueError('Nonofficial or wrong-region combination')
    if not math.isclose(sum(scores.values()), 1., abs_tol=1e-6):
        raise ValueError('Type probabilities must sum to one')
    states = json.loads(row['criterion_states'])
    eligible = {}
    excluded = []
    for combination, score in scores.items():
        if combination == 'normal' or score == 0:
            continue
        if 'spine_axis' in combination.split(';'):
            axis = states.get('spine_axis', {})
            angle = axis.get('angle_deg')
            if not (axis.get('basis') == 'measured_axis_angle' and axis.get('status') == 'fail'
                    and isinstance(angle, (int, float)) and math.isfinite(angle) and abs(angle) > AXIS_LIMIT_DEG):
                excluded.append(combination)
                continue
        eligible[combination] = score
    selected = max(sorted(eligible), key=eligible.get) if eligible else None
    mass = sum(eligible.values())
    evidence = {'basis': 'learned_type_conditional_on_existing_alarm',
                'status': 'candidate' if selected else 'undetermined',
                'selected_combination': selected, 'normal_score': scores['normal'],
                'eligible_probability_mass': mass,
                'conditional_score': eligible[selected] / mass if selected else None,
                'excluded_axis_combinations': excluded,
                'secondary_normal_disagreement': scores['normal'] >= max(eligible.values(), default=0.),
                'clinical_validation': False, 'requires_review': True,
                'score_scope': 'model distribution after conditioning; not calibrated confidence'}
    result.update(violation_type=official_violation_type(selected.split(';')) if selected else '',
                  decision_version='experimental-conditional-1',
                  decision_reason='conditional_type_candidate' if selected else 'conditional_type_unavailable',
                  conditional_type_evidence=json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    # Preserve quality score, binary alarm, physical criteria and their provenance.
    return result
