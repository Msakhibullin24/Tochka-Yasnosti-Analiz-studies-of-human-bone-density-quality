"""Versioned decision contract shared by inference and held-out validation."""
from __future__ import annotations

import math

VERSION = "5"
IMPLANT_SAT_FRAC = 0.015
AXIS_LIMIT_DEG = 5.0


def decide(group: str, score: float, criteria: dict[str, float], features: dict,
           quality_threshold: float, criterion_thresholds: dict[str, float],
           axis_rule: str = "official") -> dict:
    """Build the versioned decision.

    axis_rule selects how spine_axis is resolved:
      "official" - the measured angle against AXIS_LIMIT_DEG, as shipped;
      "model"    - the criterion score against its own validated threshold.

    "official" stays the default so inference behaviour is unchanged. "model" exists
    because the official rule is unreachable on some cohorts: measured independently
    on the organizer data, no spine frame shows a visible tilt above AXIS_LIMIT_DEG,
    so the rule fires on 0 of 10 labelled positives while discarding a classifier
    score that does separate the classes. Switching is a clinical-definition change
    and is never automatic.
    """
    if axis_rule not in ("official", "model"):
        raise ValueError("axis_rule must be 'official' or 'model'")
    if not math.isfinite(score) or any(not math.isfinite(p) for p in criteria.values()):
        raise ValueError("non-finite model score")
    states, review = {}, []
    for key, value in criteria.items():
        predicted = value >= criterion_thresholds.get(key, 0.5)
        state = {'status': 'fail' if predicted else 'pass', 'basis': 'criterion_threshold'}
        if key == 'spine_axis':
            angle = features.get('spine_abs_angle_deg')
            if axis_rule == 'model':
                state = {'status': 'fail' if predicted else 'pass', 'basis': 'criterion_threshold',
                         'angle_deg': float(abs(angle)) if angle is not None and math.isfinite(angle) else None,
                         'limit_deg': AXIS_LIMIT_DEG, 'axis_rule': 'model',
                         'clinical_validation': False}
            elif angle is None or not math.isfinite(angle):
                state = {'status': 'undetermined', 'basis': 'axis_measurement_unavailable'}
                review.append('axis_measurement_unavailable')
            else:
                measured = abs(angle) > AXIS_LIMIT_DEG
                state = {'status': 'fail' if measured else 'pass', 'basis': 'measured_axis_angle',
                         'angle_deg': float(abs(angle)), 'limit_deg': AXIS_LIMIT_DEG,
                         'clinical_validation': False}
                if measured != predicted:
                    review.append('axis_model_measurement_disagreement')
        states[key] = state
    violations = [key for key, state in states.items() if state['status'] == 'fail']
    binary_signal = score >= quality_threshold
    # Binary detection and typification are separate tasks. An unknown type must
    # never turn a detected violation into a normal image. Export the binary
    # answer unchanged; acceptance reports the missing typification explicitly.
    quality = int(bool(violations) or binary_signal)
    type_status = 'identified' if violations else 'undetermined' if binary_signal else 'not_detected'
    if type_status == 'undetermined':
        review.append('violation_type_undetermined')
    # Bright pixels alone do not prove a prosthesis or an incorrectly placed ROI.
    if group == 'hip' and features.get('sat_frac', 0.0) >= IMPLANT_SAT_FRAC:
        review.append('suspected_metal_requires_review')
    return {'score': score, 'quality': quality, 'violations': violations,
            'criterion_states': states, 'violation_type_status': type_status,
            'review_reasons': review, 'criterion_thresholds': dict(criterion_thresholds),
            'quality_threshold': quality_threshold,
            'decision_reason': 'criterion_failure' if violations else 'binary_only_review' if binary_signal else 'no_detected_violation',
            'decision_version': VERSION}
