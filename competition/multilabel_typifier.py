"""Independent official criteria with study-held-out threshold fitting.

No type is inferred from a binary alarm. Unlearnable criteria stay unavailable.
Clinical axis constraints are explicit at prediction time, never relabeled.
"""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dxaqc.model import CRITERIA, best_f1_threshold
from structured_typifier import StructuredTypifier


def fit_head(matrix, labels):
    if len(np.unique(labels)) < 2:
        return None
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=.01, class_weight='balanced', max_iter=5000, random_state=17)).fit(matrix, labels)


class MultilabelTypifier(StructuredTypifier):
    def __init__(self, group):
        super().__init__(group)

    def fit(self, features, embedding, targets, studies):
        matrix = self.matrix(features, embedding)
        studies = np.asarray(studies)
        if len(targets) != len(matrix) or studies.shape != (len(matrix),) or len(np.unique(studies)) < 3:
            raise ValueError('Need three independent training studies')
        for value in targets:
            if value != 'normal' and not set(value.split(';')) <= CRITERIA[self.group].keys():
                raise ValueError('Nonofficial criterion target')
        self.models, self.thresholds, self.inner_splits = {}, {}, []
        folds = list(GroupKFold(3).split(matrix, groups=studies))
        for train, test in folds:
            if set(studies[train]) & set(studies[test]):
                raise ValueError('Study leakage')
            self.inner_splits.append({'training_studies': sorted(set(studies[train])),
                                      'test_studies': sorted(set(studies[test]))})
        for code in CRITERIA[self.group]:
            labels = np.array([int(code in value.split(';')) for value in targets])
            scores = np.full(len(matrix), np.nan)
            for train, test in folds:
                model = fit_head(matrix[train], labels[train])
                if model is not None:
                    scores[test] = model.predict_proba(matrix[test])[:, 1]
            self.models[code] = fit_head(matrix, labels)
            valid = np.isfinite(scores)
            self.thresholds[code] = (best_f1_threshold(labels[valid], scores[valid])
                                     if self.models[code] is not None and len(np.unique(labels[valid])) == 2
                                     else float(np.nextafter(1., np.inf)))
        return self

    def predict(self, features, embedding, axis_guard=True):
        matrix = self.matrix(features, embedding)
        scores = {code: model.predict_proba(matrix)[:, 1] if model is not None else np.zeros(len(matrix))
                  for code, model in self.models.items()}
        predictions = []
        for i, geometry in enumerate(features):
            selected, conflicts = [], []
            for code in CRITERIA[self.group]:
                if scores[code][i] < self.thresholds[code]:
                    continue
                if code == 'spine_axis' and axis_guard:
                    angle = geometry.get('spine_abs_angle_deg')
                    if angle is None or not np.isfinite(angle) or abs(angle) <= 5.:
                        conflicts.append('axis_score_without_measurement_above_5deg')
                        continue
                selected.append(code)
            predictions.append({'codes': selected, 'criterion_scores': {c: float(s[i]) for c, s in scores.items()},
                                'thresholds': dict(self.thresholds), 'conflicts': conflicts,
                                'clinical_validation': False, 'basis': 'independent_learned_criteria'})
        return predictions
