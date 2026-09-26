"""Experimental coherent normal/type classifier with explicit reference filtering.

Targets are observed combinations of official criteria, including normal. The
classifier never fills a type merely because a binary score is high. Contradictory
and incomplete reference rows cannot become training targets.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_release_review import label_taxonomy_conflict
from dxaqc.model import CRITERIA, REGIONS, group_of


def target(row):
    if row.get('region') not in REGIONS:
        return None
    group = group_of(row['region'])
    if label_taxonomy_conflict(row) or any(row.get(k) not in ('0', '1') for k in CRITERIA[group]):
        return None
    if row.get('quality_class') not in ('0', '1'):
        return None
    return ';'.join(k for k in CRITERIA[group] if row[k] == '1') or 'normal'


class StructuredTypifier:
    def __init__(self, group, family='logistic'):
        if group not in CRITERIA or family not in ('logistic', 'forest'):
            raise ValueError('Unsupported typifier anatomy or model family')
        self.group, self.family = group, family
        self.feature_names = sorted({name for names in CRITERIA[group].values() for name in names})

    def matrix(self, features, embedding):
        geometry = np.array([[f.get(name, np.nan) for name in self.feature_names]
                             for f in features], dtype=float)
        embedding = np.asarray(embedding, dtype=float)
        if embedding.ndim != 2 or len(embedding) != len(features) or not np.isfinite(embedding).all():
            raise ValueError('Invalid frozen embeddings')
        return np.column_stack([geometry, embedding])

    def fit(self, features, embedding, targets):
        if len(targets) != len(features) or len(set(targets)) < 2 or 'normal' not in targets:
            raise ValueError('Typifier training requires normal and typed examples')
        for value in targets:
            if value != 'normal' and not set(value.split(';')) <= CRITERIA[self.group].keys():
                raise ValueError('Target contains a nonofficial or wrong-region criterion')
        estimator = (LogisticRegression(C=.01, class_weight='balanced', max_iter=5000, random_state=17)
                     if self.family == 'logistic' else
                     RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            class_weight='balanced_subsample', random_state=17, n_jobs=1))
        self.model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
        self.model.fit(self.matrix(features, embedding), targets)
        return self

    def predict(self, features, embedding):
        probabilities = self.model.predict_proba(self.matrix(features, embedding))
        classes = self.model.classes_
        return [{'codes': [] if classes[index] == 'normal' else classes[index].split(';'),
                 'normal_score': float(values[list(classes).index('normal')]),
                 'combination_scores': dict(zip(classes, map(float, values))),
                 'basis': 'learned_reference_combination', 'clinical_validation': False}
                for values, index in zip(probabilities, probabilities.argmax(axis=1))]
