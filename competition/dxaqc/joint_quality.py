"""A coherent quality/type distribution with an explicit physical axis constraint."""
from __future__ import annotations

import math

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .decision import AXIS_LIMIT_DEG
from .model import CRITERIA


def select_quality_threshold(truth, scores, forced):
    """Tune the actual joint decision, including mandatory measured failures."""
    truth, scores, forced = np.asarray(truth), np.asarray(scores, float), np.asarray(forced, bool)
    if truth.ndim != 1 or truth.shape != scores.shape or scores.shape != forced.shape or not len(truth):
        raise ValueError('Invalid inner-fold threshold arrays')
    if not np.isin(truth, [0, 1]).all() or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('Invalid inner-fold labels or scores')
    candidates = np.unique(np.r_[0., scores, np.nextafter(1., np.inf)])
    best, best_f1 = .5, -1.
    for threshold in candidates:
        predicted = (scores >= threshold) | forced
        tp = int(((truth == 1) & predicted).sum())
        fp = int(((truth == 0) & predicted).sum())
        fn = int(((truth == 1) & ~predicted).sum())
        f1 = 2*tp/max(1, 2*tp+fp+fn)
        if f1 > best_f1 or (f1 == best_f1 and abs(threshold-.5) < abs(best-.5)):
            best, best_f1 = float(threshold), f1
    return best


class JointQualityModel:
    def __init__(self, group, axis_policy='measured'):
        if group not in CRITERIA or axis_policy not in ('measured', 'advisory'):
            raise ValueError('Unsupported anatomy')
        self.group = group
        self.axis_policy = axis_policy
        self.feature_names = sorted({n for names in CRITERIA[group].values() for n in names})
        self.quality_threshold = .5

    def matrix(self, features, embedding):
        embedding = np.asarray(embedding, dtype=float)
        if embedding.ndim != 2 or len(features) != len(embedding) or not np.isfinite(embedding).all():
            raise ValueError('Invalid frozen embeddings')
        geometry = np.array([[f.get(n, np.nan) for n in self.feature_names] for f in features], dtype=float)
        return np.column_stack([geometry, embedding])

    def fit(self, features, embedding, targets):
        if len(targets) != len(features) or 'normal' not in targets or len(set(targets)) < 2:
            raise ValueError('Need normal and observed positive combinations')
        for target in targets:
            codes = target.split(';')
            if target != 'normal' and (not set(codes) <= CRITERIA[self.group].keys() or len(codes) != len(set(codes))):
                raise ValueError('Nonofficial or duplicate training type')
        self.model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                  LogisticRegression(C=.01, class_weight='balanced', max_iter=5000, random_state=17))
        self.model.fit(self.matrix(features, embedding), targets)
        return self

    def distributions(self, features, embedding):
        probabilities = self.model.predict_proba(self.matrix(features, embedding))
        results = []
        for geometry, values in zip(features, probabilities):
            classes = dict(zip(self.model.classes_, map(float, values)))
            allowed, excluded = {}, []
            angle = geometry.get('spine_abs_angle_deg')
            for combination, score in classes.items():
                if self.axis_policy == 'measured' and 'spine_axis' in combination.split(';') and not (
                    isinstance(angle, (int, float)) and math.isfinite(angle) and abs(angle) > AXIS_LIMIT_DEG):
                    excluded.append(combination)
                else:
                    allowed[combination] = score
            total = sum(allowed.values())
            if not total > 0 or 'normal' not in allowed:
                raise ValueError('No coherent distribution')
            allowed = {k: v/total for k, v in allowed.items()}
            results.append({'combination_scores': allowed, 'excluded_combinations': excluded,
                            'score': 1.-allowed['normal']})
        return results

    def decisions(self, features, embedding):
        decisions = []
        for geometry, distribution in zip(features, self.distributions(features, embedding)):
            positive = {k: v for k, v in distribution['combination_scores'].items() if k != 'normal'}
            quality = int(bool(positive) and distribution['score'] >= self.quality_threshold)
            codes = max(sorted(positive), key=positive.get).split(';') if quality else []
            angle = geometry.get('spine_abs_angle_deg')
            measured_axis_failure = self.axis_policy == 'measured' and self.group == 'spine' and isinstance(angle, (int, float)) and math.isfinite(angle) and abs(angle) > AXIS_LIMIT_DEG
            if measured_axis_failure and 'spine_axis' not in codes:
                codes.append('spine_axis')
            quality = int(bool(codes))
            states = {}
            for key in CRITERIA[self.group]:
                state = {'status': 'fail' if key in codes else 'pass', 'basis': 'learned_joint_combination',
                         'clinical_validation': False}
                if key == 'spine_axis':
                    angle = geometry.get('spine_abs_angle_deg')
                    state.update(angle_deg=float(angle) if isinstance(angle, (int, float)) and math.isfinite(angle) else None,
                                 limit_deg=AXIS_LIMIT_DEG)
                    if key in codes:
                        state['basis'] = 'measured_axis_angle_and_learned_type' if self.axis_policy == 'measured' else 'learned_axis_criterion'
                    elif state['angle_deg'] is None:
                        state.update(status='undetermined', basis='axis_measurement_unavailable')
                states[key] = state
            review = ['joint_model_requires_clinical_validation']
            if distribution['excluded_combinations']:
                review.append('joint_axis_probability_excluded_by_geometry')
            if self.group == 'spine' and self.axis_policy == 'advisory':
                measured = isinstance(angle, (int, float)) and math.isfinite(angle) and abs(angle) > AXIS_LIMIT_DEG
                if measured != ('spine_axis' in codes):
                    review.append('axis_model_measurement_disagreement')
            decisions.append({'score': distribution['score'], 'quality': quality, 'violations': codes,
                              'criterion_states': states, 'violation_type_status': 'identified' if codes else 'not_detected',
                              'review_reasons': review, 'criterion_thresholds': {},
                              'quality_threshold': self.quality_threshold,
                              'decision_reason': 'joint_typed_violation' if codes else 'joint_normal',
                              'decision_version': 'joint-1', 'joint_evidence': distribution})
        return decisions

    def review_untyped(self, decision, features, embedding):
        """Resolve only binary-only alarms with a supervised normal/type model."""
        if not decision['quality'] or decision['violations']:
            return decision
        values = self.model.predict_proba(self.matrix([features], embedding))[0]
        probabilities = dict(zip(self.model.classes_, map(float, values)))
        combination = str(self.model.classes_[int(values.argmax())])
        codes = [] if combination == 'normal' else combination.split(';')
        states = {k: dict(v) for k, v in decision['criterion_states'].items()}
        for key in states:
            if key in codes:
                states[key].update(status='fail', basis='supervised_untyped_review', clinical_validation=False)
        review = [r for r in decision['review_reasons'] if r != 'violation_type_undetermined']
        review.append('secondary_supervised_review')
        angle = features.get('spine_abs_angle_deg')
        if 'spine_axis' in codes and (angle is None or not math.isfinite(angle) or abs(angle) <= AXIS_LIMIT_DEG):
            review.append('axis_model_measurement_disagreement')
        return {**decision, 'score': 1.-probabilities['normal'], 'quality': int(bool(codes)),
                'violations': codes, 'criterion_states': states,
                'violation_type_status': 'identified' if codes else 'not_detected',
                'review_reasons': review, 'decision_version': 'review-1',
                'decision_reason': 'supervised_type_review' if codes else 'supervised_normal_review',
                'joint_evidence': {'basis': 'observed_reference_combination', 'combination_scores': probabilities,
                                   'clinical_validation': False, 'original_score': decision['score']}}
