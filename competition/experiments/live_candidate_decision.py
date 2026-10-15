"""Inference decisions for saved gated heads and independent fold voting.

A detected overall defect cannot disappear because members disagree on type.
Conditional type predictions remain explicitly marked for review; the physical
spine-axis limit is never replaced by argmax or a majority classifier vote.
"""
from __future__ import annotations

import numpy as np

from dxaqc.decision import decide
from experiments.quality_gated_decoder import gated_decisions


def gated_result(group, criteria, features, quality, saved):
    codes = list(criteria)
    spec = saved['criterion_decoder']
    thresholds = spec['thresholds']
    angle = features.get('spine_abs_angle_deg')
    selected, conditional = gated_decisions([[criteria[c] for c in codes]], [thresholds[c] for c in codes], codes,
        np.array([angle if angle is not None else np.nan]), [quality], saved['quality_cut'], saved.get('preserve_independent', False))
    result = decide(group, quality, criteria, features, saved['quality_cut'], thresholds)
    for j, code in enumerate(codes):
        if code == 'spine_axis':
            continue
        result['criterion_states'][code] = {'status': 'fail' if selected[0, j] else 'pass',
            'basis': 'quality_conditioned_argmax' if conditional[0] and selected[0, j] else
                'criterion_threshold_preserved' if saved.get('preserve_independent', False) else 'quality_gated_criterion',
            'criterion_score': criteria[code], 'independent_threshold': float(thresholds[code]), 'clinical_validation': False}
    result['violations'] = [code for j, code in enumerate(codes) if selected[0, j]]
    result['quality'] = int(bool(result['violations']))
    result['violation_type_status'] = 'identified' if result['violations'] else 'not_detected'
    result['review_reasons'] = [r for r in result['review_reasons'] if r != 'violation_type_undetermined']
    if conditional[0]:
        result['review_reasons'].append('conditional_type_prediction_requires_review')
    result['decision_reason'] = 'quality_conditioned_argmax_review' if conditional[0] else 'quality_gated_types'
    result['decision_version'] = 'research-gated-live-1'
    return result


def ensemble_result(group, decisions, raw_scores, quality, features):
    codes = list(raw_scores[0])
    votes = {c: float(np.mean([c in d['violations'] for d in decisions])) for c in codes}
    result = decide(group, quality, votes, features, float(np.nextafter(1., np.inf)), {c: .5 for c in codes})
    quality_vote = float(np.mean([d['quality'] for d in decisions]))
    if quality_vote >= .5 and not result['violations']:
        allowed = [c for c in codes if c != 'spine_axis']
        if not allowed:
            raise ValueError('A non-axis type is required for conditional ensemble typing')
        selected = max(allowed, key=lambda c: (votes[c], np.mean([s[c] for s in raw_scores]), -codes.index(c)))
        result['violations'] = [selected]
        result['quality'] = 1
        result['violation_type_status'] = 'identified'
        result['criterion_states'][selected] = {'status': 'fail', 'basis': 'ensemble_quality_conditioned_argmax',
            'criterion_vote': votes[selected], 'quality_vote': quality_vote, 'clinical_validation': False}
        result['review_reasons'].append('ensemble_type_disagreement_requires_review')
        result['decision_reason'] = 'ensemble_type_disagreement_review'
    result['ensemble_quality_vote'] = quality_vote
    return result, votes
