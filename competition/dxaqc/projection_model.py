"""Offline pixel-only spine view prediction from a pinned public-source model."""
import hashlib
from pathlib import Path

import cv2
import joblib
import numpy as np

from .embedding import embed, WEIGHTS_SHA256


class ProjectionModel:
    def __init__(self, path):
        path = Path(path)
        bundle = joblib.load(path)
        if (bundle.get('schema_version') != 1 or bundle.get('encoder_sha256') != WEIGHTS_SHA256
                or bundle.get('clinical_validation') is not False):
            raise ValueError('Incompatible projection model')
        self.model = bundle['model']
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def predict(self, pixels, region):
        if region != 'spine':
            return None
        square = cv2.resize(pixels, (320, 320), interpolation=cv2.INTER_AREA)
        vectors = np.stack([embed(square), embed(255-square)])
        score = float(self.model.predict_proba(vectors)[:, 1].mean())
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid projection score')
        value = 'frontal' if score <= .1 else 'lateral' if score >= .9 else 'unknown'
        return {'value': value, 'status': 'candidate' if value != 'unknown' else 'undetermined',
                'lateral_score': score, 'method': 'learned_pixel_only_projection',
                'model_sha256': self.sha256, 'clinical_validation': False,
                'model_affects_decision': False, 'reason': 'Source-label classifier; clinical view verification unavailable.'}
