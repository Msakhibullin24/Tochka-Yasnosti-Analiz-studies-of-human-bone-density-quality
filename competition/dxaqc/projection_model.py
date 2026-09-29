"""Offline pixel-only spine view prediction from a pinned public-source model."""
import hashlib
from pathlib import Path

import cv2
import joblib
import numpy as np

from .embedding import embed, WEIGHTS_SHA256


def require_supported_projection(projection):
    """Withhold frontal QC for an unsupported or uncertain learned view.

    This is an inference scope gate, not a claim of clinical validation.
    A missing prediction (hip models currently return None) is kept distinct
    from an evaluated but uncertain spine projection.
    """
    if projection is None:
        return
    from .dicom_io import DicomReadError
    value = projection.get('value')
    if value == 'lateral':
        raise DicomReadError('UNSUPPORTED_PROJECTION',
                             'pixel-based model detected a lateral view; frontal quality assessment withheld')
    if value != 'frontal':
        raise DicomReadError('UNCERTAIN_PROJECTION',
                             'pixel-based projection is uncertain; frontal quality assessment withheld')


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
                'model_affects_decision': True,
                'decision_scope': 'frontal_qc_eligibility',
                'reason': 'Проекция определяет допуск к анализу прямого снимка; точность модели требует отдельной проверки.'}
