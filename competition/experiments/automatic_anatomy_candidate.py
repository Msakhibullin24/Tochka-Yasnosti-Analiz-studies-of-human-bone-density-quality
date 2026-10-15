"""Add explicit anatomical measurements without changing fitted QC inputs.

Masks are captured from the existing prediction calls, not recomputed. The
new checks expose model-based findings and uncertainty, never clinical truth.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path

import numpy as np

from experiments.live_anatomy_candidate import LiveAnatomyCandidate
from experiments.automatic_anatomy_geometry import th12_visibility, iliac_crests, hip_roi_relationships, MUSCLE_CHANNELS
from experiments.mask_anatomy_features import largest_component
from gpu_research.common import digest


class CapturedPredictor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.history = deque(maxlen=2)

    def predict_single_npy_array(self, *args, **kwargs):
        output = self.predictor.predict_single_npy_array(*args, **kwargs)
        self.history.append(np.asarray(output).squeeze().copy())
        return output

    def __getattr__(self, name):
        return getattr(self.predictor, name)


class AutomaticAnatomyCandidate(LiveAnatomyCandidate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anatomy = CapturedPredictor(self.anatomy)
        if self.pelvis is not None: self.pelvis = CapturedPredictor(self.pelvis)
        self.geometry_records = []
        self.geometry_code_sha256 = digest(Path(__file__).with_name('automatic_anatomy_geometry.py'))
        self.workflow_profile.update(automatic_geometry=True, geometry_code_sha256=self.geometry_code_sha256,
            anatomical_evidence_scope='model-derived measurements and explicit uncertainty, not clinical validation')

    def hip_anatomy(self, image, region, features, result, **kwargs):
        masks = super().hip_anatomy(image, region, features, result, **kwargs)
        selected = masks.copy()
        branch = 'native'
        if len(self.pelvis.history) == 2:
            first, flipped = self.pelvis.history
            other = flipped[..., ::-1]
            if largest_component(other[6] | other[7]).sum() > largest_component(first[6] | first[7]).sum():
                selected[list(MUSCLE_CHANNELS)] = other[list(MUSCLE_CHANNELS)]
                branch = 'mirrored_then_inverse_mapped'
        learned = result['learned_anatomy']
        learned['computed_geometry'] = hip_roi_relationships(selected, learned.get('landmark_points'),
            learned['neck_roi_assessment'], image.pixel_mm_x, image.pixel_mm)
        learned['computed_geometry']['muscle_model_input_branch'] = branch
        return masks

    def _analyze(self, image):
        self.anatomy.history.clear()
        if self.pelvis is not None: self.pelvis.history.clear()
        result = super()._analyze(image)
        if result['region'] == 'spine':
            geometry = {'th12': th12_visibility(self.anatomy.history[-1], image.pixel_mm_x, image.pixel_mm),
                        'iliac_crests': iliac_crests(self.pelvis.history[-1]) if self.pelvis is not None else []}
            result['learned_anatomy']['computed_geometry'] = geometry
        else:
            geometry = result['learned_anatomy']['computed_geometry']
        geometry['provider_code_sha256'] = self.geometry_code_sha256
        geometry['clinical_validation'] = False
        geometry['affects_qc_classifier'] = False
        result['joint_evidence']['computed_anatomical_geometry'] = {
            'provider_code_sha256': self.geometry_code_sha256, 'qc_inputs_changed': False,
            'anatomical_truth_verified': False}
        self.geometry_records.append({'decoded_pixel_sha256': image.pixel_sha256, 'region': result['region'],
            'physical_scale_source': image.pixel_mm_source, 'geometry': geometry})
        return result
