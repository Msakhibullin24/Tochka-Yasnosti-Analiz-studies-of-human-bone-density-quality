"""Raw-DICOM inference for the measured anatomical QC research candidate.

Uses saved heads/decoders only: no fitting, labels or cached image features at
inference. Study-held-out replay is explicit. Deployment banks identify their
full-data fit or fold-ensemble strategy; neither is scored on its training data.
This does not replace the released profile.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import socket
import warnings
import hashlib
import copy
from threading import RLock

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.decision import decide
from dxaqc.learned_anatomy import mask_regions, roi_proposals
from dxaqc.model import _matrix, group_of
from dxaqc.pipeline import Analyzer, Options, run_batch
from experiments.mask_anatomy_features import spine_features, mask_statistics
from experiments.anatomical_visibility_rule import apply_visibility
from experiments.numbered_spine_roi import propose_spine_rois
from experiments.hip_mask_features import hip_measurements
from experiments.mask_anatomy_features import largest_component
from experiments.train_silhouette_heatmaps import Heatmaps
from experiments.train_camera_projection import ProjectionNet
from experiments.pelvis_camera_projection import normalized_channels
from experiments.neck_roi_from_mask import neck_roi
from experiments.adapt_native_neck_roi import anatomical_frame, proposal_from_template
from experiments.roi_candidate_features import measure_roi
from experiments.live_candidate_decision import gated_result, ensemble_result
from experiments.three_iteration_upgrade import predict_family, decode_scores
from gpu_research.common import digest, image_input, load_encoder, representations

EXECUTION_SOURCE = Path(__file__).read_bytes()
EXECUTION_SHA256 = hashlib.sha256(EXECUTION_SOURCE).hexdigest()


def anatomical_axis_features(features, angle, enabled):
    updated = dict(features)
    if enabled:
        updated['spine_abs_angle_deg'] = abs(float(angle)) if angle is not None else None
    return updated


def load_bank(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if digest(directory / 'bank.joblib') != manifest['bank_sha256']:
        raise ValueError('Fitted head-bank checksum differs')
    value = joblib.load(directory / 'bank.joblib')
    folds = {(s['fold'], s['group']): {k: s[k] for k in ('fold', 'group', 'heads', 'columns', 'codes')}
             for s in value['folds']}
    # Keep only study->fold identities for evaluation. No QC labels, inner OOF
    # scores or reference image features are retained in the inference object.
    studies = {} if value.get('inference_only') else dict(zip(value['baseline'].study.astype(str), value['baseline'].fold.astype(int)))
    return manifest, folds, studies


class LiveAnatomyCandidate:
    def __init__(self, bank, ranking_bank, decoders, feature_spec, anatomy_model,
                 encoder_directories, *, held_out_replay=False, replay_manifest=None,
                 pelvis_model=None, pelvis_checkpoint_sha256=None,
                 hip_landmarks=None, hip_projection=None, neck_template=None, visibility_cue=True,
                 hip_display_landmarks=None, hip_display_template=None):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        self.base = Analyzer()
        self.specialist = self.base.specialist
        self.specialist_portfolio = self.base.specialist_portfolio
        self.inference_lock = RLock()
        self.bundle = copy.copy(self.base.bundle)
        self.held_out_replay = held_out_replay
        self.visibility_cue = visibility_cue
        self.manifest, self.folds, self.source_groups = load_bank(bank)
        if held_out_replay and self.manifest.get('full_data_fit'):
            raise ValueError('A full-data trained deployment cannot be used for held-out replay')
        self.replay_manifest = None
        if held_out_replay:
            if replay_manifest is None:
                raise ValueError('Held-out replay requires an audited DICOM identity manifest')
            replay = json.loads(replay_manifest.read_text())
            if (replay.get('schema_version') != 1 or replay.get('bank_sha256') != self.manifest['bank_sha256'] or
                    replay.get('source_fingerprint') != self.manifest['source_fingerprint'] or
                    replay.get('cross_group_identity_collisions') != 0):
                raise ValueError('Replay source identity differs')
            self.replay_manifest = replay
        old_manifest, self.reference, _ = load_bank(ranking_bank)
        if any(self.manifest[k] != old_manifest[k] for k in ('labels_sha256', 'baseline_sha256', 'source_fingerprint')):
            raise ValueError('Quality-reference identities differ')
        evaluation = json.loads((decoders / 'evaluation.json').read_text())
        if evaluation['bank_sha256'] != self.manifest['bank_sha256'] or evaluation['ranking_bank_sha256'] != old_manifest['bank_sha256']:
            raise ValueError('Decoder head-bank identity differs')
        if evaluation.get('decoder_sha256') is not None and digest(decoders / 'decoders.joblib') != evaluation['decoder_sha256']:
            raise ValueError('Exported decoder checksum differs')
        self.decoders = {(s['fold'], s['group']): s for s in joblib.load(decoders / 'decoders.joblib')}
        self.gated = all('criterion_decoder' in s for s in self.decoders.values())
        if not self.gated and any('decoder' not in s or s['decoder']['calibrator'] is not None for s in self.decoders.values()):
            raise ValueError('Unsupported typed decoder representation')
        if self.gated and any(s['criterion_decoder']['exponent'] != 0 or s['criterion_decoder']['calibrator'] is not None
                              for s in self.decoders.values()):
            raise ValueError('Gated inference requires the measured independent typed thresholds')
        self.workflow_profile = {'profile_id': 'research-live-anatomy-qc-20261015', 'clinical_validation': False,
                                 'status': 'research_only', 'source_profile_id':
                                 (self.base.workflow_profile or {}).get('profile_id'),
                                 'evaluation_mode': 'study_held_out_replay' if held_out_replay else
                                     'full_data_fit' if self.manifest.get('full_data_fit') else 'fold_ensemble'}
        self.workflow_profile.update(gated_decoder=self.gated, spine_visibility_cue=visibility_cue)
        self.records = []
        extra = json.loads(feature_spec.read_text())
        if digest(feature_spec) != self.manifest['extra_features_sha256']:
            raise ValueError('Anatomical feature specification differs')
        self.hip_features_used = bool(extra['columns_by_group']['hip'])
        if self.hip_features_used:
            if (hip_landmarks is None or neck_template is None or
                    extra['hip_checkpoint_sha256'] != pelvis_checkpoint_sha256 or
                    extra['hip_landmark_checkpoint_sha256'] != digest(hip_landmarks) or
                    extra['native_roi_template_sha256'] != digest(neck_template) or
                    extra['roi_geometry_code_sha256'] != digest(Path(__file__).with_name('roi_candidate_features.py'))):
                raise ValueError('Hip feature producer/model identities differ')
        if digest(anatomy_model / 'fold_0/checkpoint_best.pth') != extra['checkpoint_sha256']:
            raise ValueError('Anatomy checkpoint differs from feature generation')
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        self.anatomy = nnUNetPredictor(use_mirroring=False, allow_tqdm=False, verbose=False,
                                      device=torch.device('cuda'))
        self.anatomy.initialize_from_trained_model_folder(str(anatomy_model), use_folds=(0,),
                                                         checkpoint_name='checkpoint_best.pth')
        self.anatomy_sha256 = extra['checkpoint_sha256']
        self.pelvis = None
        self.pelvis_sha256 = pelvis_checkpoint_sha256
        if pelvis_model is not None:
            if digest(pelvis_model / 'fold_0/checkpoint_best.pth') != pelvis_checkpoint_sha256:
                raise ValueError('Pelvic anatomy checkpoint checksum differs')
            self.pelvis = nnUNetPredictor(use_mirroring=False, allow_tqdm=False, verbose=False,
                                        device=torch.device('cuda'))
            self.pelvis.initialize_from_trained_model_folder(str(pelvis_model), use_folds=(0,),
                                                           checkpoint_name='checkpoint_best.pth')
        self.hip_landmarks = self.hip_projection = None
        self.hip_model_hashes = {}
        self.neck_template = None
        self.hip_display_model = self.hip_display_template = None
        if hip_landmarks is not None or hip_projection is not None:
            if pelvis_model is None or hip_landmarks is None or hip_projection is None:
                raise ValueError('Hip anatomy needs pelvis masks, landmarks and projection models together')
            landmarks = torch.load(hip_landmarks, map_location='cpu', weights_only=True)
            projection = torch.load(hip_projection, map_location='cpu', weights_only=True)
            if (landmarks['normalization'] != 'physical_proximal_mask150mm_10mm_pad' or
                    projection['channels'] != ['femur', 'ipsilateral_pelvis']):
                raise ValueError('Unexpected hip model representation')
            self.hip_landmarks = Heatmaps(landmarks.get('input_channels', 1)).cuda().eval()
            self.hip_projection = ProjectionNet().cuda().eval()
            self.hip_landmarks.load_state_dict(landmarks['state_dict'], strict=True)
            self.hip_projection.load_state_dict(projection['state_dict'], strict=True)
            self.hip_model_hashes = {'landmarks': digest(hip_landmarks), 'projection': digest(hip_projection)}
        if neck_template is not None:
            template = json.loads(neck_template.read_text())
            if template['landmark_checkpoint_sha256'] != self.hip_model_hashes.get('landmarks'):
                raise ValueError('Native ROI template landmark checkpoint differs')
            self.neck_template = np.asarray(template['parameters'], float)
            self.hip_model_hashes['neck_template'] = digest(neck_template)
        if hip_display_landmarks is not None or hip_display_template is not None:
            if self.hip_landmarks is None or hip_display_landmarks is None or hip_display_template is None:
                raise ValueError('Display refinement requires primary hip models, a display network and its ROI template')
            display = torch.load(hip_display_landmarks, map_location='cpu', weights_only=True)
            template = json.loads(hip_display_template.read_text())
            if (display['normalization'] != 'physical_proximal_mask150mm_10mm_pad' or
                    template['landmark_checkpoint_sha256'] != digest(hip_display_landmarks)):
                raise ValueError('Display landmark/ROI template representation differs')
            self.hip_display_model = Heatmaps(display.get('input_channels', 1)).cuda().eval()
            self.hip_display_model.load_state_dict(display['state_dict'], strict=True)
            self.hip_display_template = np.asarray(template['parameters'], float)
            self.hip_model_hashes.update(display_landmarks=digest(hip_display_landmarks),
                                         display_template=digest(hip_display_template))
        self.encoders = []
        for directory, embedding in zip(encoder_directories, self.manifest['embeddings'], strict=True):
            metadata = embedding.get('metadata')
            if metadata is None:
                metadata = json.loads(Path(embedding['path']).with_suffix('.json').read_text())
            for name, checksum in metadata['weights'].items():
                if digest(directory / name) != checksum:
                    raise ValueError('Frozen encoder checkpoint differs')
            if (digest(directory / 'config.json') != metadata['config_sha256'] or
                    digest(directory / 'preprocessor_config.json') != metadata['processor_sha256']):
                raise ValueError('Frozen encoder configuration/processor differs')
            kind = json.loads((directory / 'config.json').read_text())['model_type']
            processor, encoder = load_encoder(directory, kind, 'cuda', metadata['precision'])
            self.encoders.append((processor, encoder, kind, metadata['size']))
        torch.set_num_threads(4)
        self.workflow_profile.update(inference_code_sha256=EXECUTION_SHA256,
            bank_sha256=self.manifest['bank_sha256'], reference_bank_sha256=old_manifest['bank_sha256'],
            decoder_sha256=digest(decoders / 'decoders.joblib'), hip_models=self.hip_model_hashes)
        self.bundle.meta = {'model_type': 'anatomical_qc_gated' if self.gated else 'anatomical_qc_typed_union',
            'training_scope': self.workflow_profile['evaluation_mode'], 'clinical_validation': False,
            'labels_sha256': self.manifest['labels_sha256'], 'qc_bank_sha256': self.manifest['bank_sha256'],
            'reference_bank_sha256': old_manifest['bank_sha256'], 'decoder_sha256': self.workflow_profile['decoder_sha256'],
            'calibration_version': self.base.bundle.meta.get('calibration_version'), 'source_baseline_model': self.base.bundle.meta}

    @staticmethod
    def scores(state, features, embeddings):
        return {code: predict_family(models, _matrix([features], state['columns'][code]), embeddings, [0])
                for code, models in state['heads'].items()}

    def analyze(self, image):
        with self.inference_lock:
            return self._analyze(image)

    def _analyze(self, image):
        result = self.base.analyze(image)
        features = dict(result['features'])
        reference_features = dict(features)
        displayed_numbered_angle = None
        hip_masks = None
        region, group = result['region'], group_of(result['region'])
        if group == 'spine':
            masks = np.asarray(self.anatomy.predict_single_npy_array(image.pixels[None, None].astype(np.float32),
                {'spacing': (999 * max(image.pixel_mm_x, image.pixel_mm), image.pixel_mm, image.pixel_mm_x)})).squeeze()
            extra, angle = spine_features(masks, image.pixels, image.pixel_mm_x, image.pixel_mm)
            features.update(extra)
            features = anatomical_axis_features(features, angle, self.manifest.get('numbered_anatomy_axis_used', False))
            if self.manifest.get('numbered_anatomy_axis_used', False) and angle is not None:
                displayed_numbered_angle = angle
                centers = np.array([[extra[f'anat_L{i}_center_x'] * image.pixels.shape[1],
                                     extra[f'anat_L{i}_center_y'] * image.pixels.shape[0]] for i in range(1, 5)])
                physical = centers * [image.pixel_mm_x, image.pixel_mm]
                slope, intercept = np.polyfit(physical[:, 1], physical[:, 0], 1)
                ys = centers[[0, -1], 1]
                axis = np.column_stack([(slope * ys * image.pixel_mm + intercept) / image.pixel_mm_x, ys])
                result['overlay'] = {**result['overlay'], 'axis': axis.tolist(), 'centerline': centers.tolist()}
            if self.pelvis is not None:
                pelvic_masks = np.asarray(self.pelvis.predict_single_npy_array(image.pixels[None, None].astype(np.float32),
                    {'spacing': (999 * max(image.pixel_mm_x, image.pixel_mm), image.pixel_mm, image.pixel_mm_x)})).squeeze()
                for name, channel in [('iliac_left', 8), ('iliac_right', 9)]:
                    stats, _ = mask_statistics(pelvic_masks[channel], image.pixels, image.pixel_mm_x, image.pixel_mm)
                    features.update({f'anat_{name}_{key}': value for key, value in stats.items()})
            regions = []
            for channel, name in [(7, 'Th12'), (6, 'L1'), (5, 'L2'), (4, 'L3'), (3, 'L4'), (2, 'L5')]:
                for item in mask_regions(masks[channel].astype(np.uint8)):
                    item['name'] = name
                    regions.append(item)
            spinal_roi = propose_spine_rois(masks, image.pixels, image.pixel_mm_x, image.pixel_mm)
            from experiments.anatomy_display_selection import complete_spine_roi_proposals
            contour_proposals = roi_proposals(regions)
            result['learned_anatomy'] = {'status': 'evaluated', 'method': 'pretrained_tsxr_named_masks',
                'regions': regions, 'roi_proposals': complete_spine_roi_proposals(spinal_roi['proposals'], contour_proposals),
                'model_sha256': self.anatomy_sha256,
                'density_based_roi': spinal_roi, 'contour_roi_proposals': contour_proposals,
                'coordinate_units': 'original_image_pixels', 'clinical_validation': False, 'affects_decision': True,
                'target_semantics': 'projected anatomical bone contours; no confirmed disc boundaries'}
        elif self.hip_landmarks is not None:
            hip_masks = self.hip_anatomy(image, region, features, result)
        canonical = np.ascontiguousarray(image.pixels[:, ::-1]) if region == 'hip_left' else image.pixels
        embeddings = []
        with torch.inference_mode():
            for processor, encoder, kind, size in self.encoders:
                values, _ = image_input(processor, canonical, size, kind)
                vector, _ = representations(encoder, values, kind, 'mean_global')
                embeddings.append(vector.float().cpu().numpy())
        known_fold = None
        if self.held_out_replay:
            by_pixel = self.replay_manifest['decoded_pixel_to_fold'].get(image.pixel_sha256)
            by_study = self.replay_manifest['dicom_study_to_fold'].get(image.study_uid)
            if by_pixel is not None and by_study is not None and by_pixel != by_study:
                raise ValueError('Replay pixel and DICOM study folds conflict')
            known_fold = by_pixel if by_pixel is not None else by_study
            if known_fold is None:
                raise ValueError('Unregistered study in strict held-out replay; use ensemble mode for new studies')
        folds = [known_fold] if known_fold is not None else sorted({f for f, g in self.folds if g == group})
        decisions, probabilities, raw_per_fold = [], [], []
        for fold in folds:
            state, old = self.folds[fold, group], self.reference[fold, group]
            saved = self.decoders[fold, group]
            raw = self.scores(state, features, embeddings)
            if self.gated:
                reference = self.scores(old, reference_features, embeddings)
                _, quality = decode_scores(saved['quality_decoder'], old['codes'],
                    np.column_stack([reference[c] for c in old['codes']]), reference['quality'])
                actual = {code: float(raw[code][0]) for code in state['codes']}
                decision = gated_result(group, actual, features, float(quality[0]), saved)
            else:
                spec, ranking = saved['decoder'], saved['ranking_blend']
                scores, quality = decode_scores(spec, state['codes'], np.column_stack([raw[c] for c in state['codes']]), raw['quality'])
                if ranking is not None:
                    reference = self.scores(old, reference_features, embeddings)
                    _, old_quality = decode_scores(ranking['prior_decoder'], old['codes'],
                        np.column_stack([reference[c] for c in old['codes']]), reference['quality'])
                    quality = ranking['current_weight'] * quality + (1 - ranking['current_weight']) * old_quality
                actual = dict(zip(state['codes'], scores[0].tolist()))
                decision = decide(group, float(quality[0]), actual, features, float(np.nextafter(1., np.inf)), spec['thresholds'])
            decisions.append(decision)
            probabilities.append(float(quality[0]))
            raw_per_fold.append(actual)
        if len(decisions) == 1:
            decision = decisions[0]
            reported_scores = raw_per_fold[0]
        else:
            # Ensemble votes by type; the measured axis is identical in every
            # member. This ensemble is not scored on its training studies.
            decision, reported_scores = ensemble_result(group, decisions, raw_per_fold, float(np.mean(probabilities)), features)
        result.update(decision)
        if self.pelvis is not None and self.visibility_cue:
            result = apply_visibility(result, features, group)
        result['criteria'] = reported_scores
        result['research_ensemble'] = {'members': len(decisions), 'criterion_score_basis':
            'fold_violation_vote' if len(decisions) > 1 else
                'single_held_out_fold_score' if self.held_out_replay else 'single_deployment_model_score',
            'overall_quality_vote': float(np.mean([d['quality'] for d in decisions])),
            'fold_quality_thresholds': [d['quality_threshold'] for d in decisions],
            'mean_raw_scores': {c: float(np.mean([s[c] for s in raw_per_fold])) for c in raw_per_fold[0]}}
        result['joint_evidence'] = {**(result.get('joint_evidence') or {}), 'research_ensemble': result['research_ensemble']}
        if hip_masks is not None and self.hip_display_model is not None:
            qc_anatomy = result['learned_anatomy']
            refined = {}
            self.hip_anatomy(image, region, {}, refined, landmark_model=self.hip_display_model,
                template_override=self.hip_display_template, masks_override=hip_masks, affects_decision=False,
                landmark_sha256=self.hip_model_hashes['display_landmarks'])
            from experiments.anatomy_display_selection import prefer_refinement
            use_refinement = prefer_refinement(qc_anatomy, refined['learned_anatomy'])
            if use_refinement:
                result.update(refined)
            result['joint_evidence']['display_refinement'] = {
                'used': use_refinement, 'selection_basis': 'geometry_availability_only; no QC labels or target ROI',
                'refined_roi_status': refined['learned_anatomy']['neck_roi_assessment']['status'],
                'refined_roi_reason': refined['learned_anatomy']['neck_roi_assessment'].get('reason'),
                'retained_primary_reason': None if use_refinement else 'refinement_failed_but_primary_geometry_available'}
            result['joint_evidence']['qc_input_anatomy'] = {'landmark_model_sha256': self.hip_model_hashes['landmarks'],
                'physical_measurements': qc_anatomy['physical_measurements'], 'neck_roi': qc_anatomy['neck_roi_assessment'],
                'role': 'fixed QC inputs; display refinement is not fed back into fitted heads'}
        # Update presentation only after every fitted head has consumed exactly
        # the geometry used in its training bank. The reference keeps its own input.
        if displayed_numbered_angle is not None:
            features['historical_spine_angle_deg'] = features.get('spine_angle_deg')
            features['spine_angle_deg'] = displayed_numbered_angle
        if group == 'hip' and self.hip_landmarks is not None:
            history = {}
            display_measurements = result['learned_anatomy']['physical_measurements']
            for name in ('troch_top_margin_mm', 'ischium_bottom_margin_mm', 'lateral_margin_mm'):
                if 'anat_' + name in display_measurements:
                    history[name] = features.get(name)
                    features[name] = display_measurements['anat_' + name]
            result['joint_evidence'] = {**(result.get('joint_evidence') or {}),
                'frozen_model_input_measurements': history, 'display_measurement_basis': 'predicted_anatomical_masks_and_landmarks'}
        result['features'] = features
        result['decision_version'] = 'research-live-anatomy-qc-2'
        result['workflow_profile_id'] = self.workflow_profile['profile_id']
        if self.held_out_replay:
            self.records.append({'decoded_pixel_sha256': image.pixel_sha256, 'study': image.study_uid,
                             'region': region, 'folds': folds, 'score': result['score'], 'quality': result['quality'],
                             'violations': result['violations'], 'criterion_scores': result['criteria'],
                             'criterion_states': result['criterion_states'], 'criterion_thresholds': result['criterion_thresholds']})
        return result

    def hip_anatomy(self, image, region, features, result, *, landmark_model=None, template_override=None,
                    masks_override=None, affects_decision=None, landmark_sha256=None):
        if masks_override is None:
            variants = []
            for flip in (False, True):
                pixels = np.ascontiguousarray(image.pixels[:, ::-1]) if flip else image.pixels
                masks = np.asarray(self.pelvis.predict_single_npy_array(pixels[None, None].astype(np.float32),
                    {'spacing': (999 * max(image.pixel_mm_x, image.pixel_mm), image.pixel_mm, image.pixel_mm_x)})).squeeze()
                variants.append(masks[..., ::-1].copy() if flip else masks)
            masks = variants[0].copy()
            for a, b in ((6, 7), (8, 9)):
                if largest_component(variants[1][a] | variants[1][b]).sum() > largest_component(masks[a] | masks[b]).sum():
                    masks[a], masks[b] = variants[1][a], variants[1][b]
        else:
            masks = masks_override
        model = self.hip_landmarks if landmark_model is None else landmark_model
        template = self.neck_template if template_override is None else template_override
        extra, points = hip_measurements(masks, image.pixels, image.pixel_mm_x, image.pixel_mm, model)
        features.update(extra)
        femur, pelvis = largest_component(masks[6] | masks[7]), largest_component(masks[8] | masks[9])
        regions = []
        for name, mask in [('femur', femur), ('hemipelvis', pelvis)]:
            for item in mask_regions(mask.astype(np.uint8)):
                item['name'] = name
                regions.append(item)
        proposal = {'status': 'unavailable', 'reason': 'landmarks_unavailable'}
        fixed_proposal = adapted_proposal = None
        if points is not None:
            proposal = neck_roi(femur, points[2], points[3], image.pixel_mm_x, image.pixel_mm)
            fixed_proposal = proposal
            if template is not None and fixed_proposal['status'] == 'proposed':
                center, frame, length = anatomical_frame(points[2], points[3], image.pixel_mm_x, image.pixel_mm)
                adapted_proposal = proposal_from_template(template, center, frame, length,
                    image.pixel_mm_x, image.pixel_mm, image.pixels.shape)
                if adapted_proposal['status'] == 'proposed':
                    proposal = adapted_proposal
            channels = normalized_channels(femur.astype(np.uint8), pelvis.astype(np.uint8), image.pixel_mm_x, image.pixel_mm)
            if region == 'hip_right':
                channels = channels[..., ::-1].copy()
            with torch.inference_mode():
                probability = float(self.hip_projection(torch.tensor(channels[None, :2].copy(), dtype=torch.float32,
                                                      device='cuda')).softmax(1)[0, 1])
            result['learned_projection'] = {'status': 'candidate', 'value': 'lateral' if probability >= .5 else 'frontal',
                'method': 'pelvis_referenced_synthetic_camera_cnn', 'lateral_endpoint_probability': probability,
                'model_sha256': self.hip_model_hashes['projection'], 'clinical_validation': False,
                'reason': 'Оценка по синтетическим фронтальным/боковым силуэтам; точность на DXA не установлена.'}
            landmarks = []
            for name, selected in [('greater_trochanter', points[:1]), ('lesser_trochanter', points[1:2]),
                                   ('femoral_neck', points[2:4])]:
                valid = bool(((selected >= 0) & (selected < image.pixels.shape[::-1])).all())
                landmarks.append({'name': name, 'points': selected.tolist() if valid else [],
                    'status': 'candidate' if valid else 'not_localized', 'verified': False,
                    'method': 'synthetic_heatmap_on_predicted_femur_mask'})
            py, px = np.where(pelvis)
            inferior = [[float(np.mean(px[py == py.max()])), float(py.max())]] if len(py) else []
            landmarks.append({'name': 'ischium', 'points': inferior, 'verified': False,
                'status': 'candidate' if inferior else 'not_localized', 'method': 'pelvic_mask_inferior_extent_candidate'})
            result['anatomy_candidates'] = {'status': 'needs_validation', 'landmarks': landmarks,
                'coordinate_system': 'original_pixel_centres', 'clinical_validation': False,
                'learned_model_sha256': self.pelvis_sha256}
            from dxaqc.anatomy import assess_projection
            result['projection_pixels'] = assess_projection(image.pixels, region, result['anatomy_candidates'])
        roi_features, roi_geometry = measure_roi(proposal, femur, pelvis, points, image.pixel_mm_x, image.pixel_mm)
        features.update(roi_features)
        extra.update(roi_features)
        result['learned_anatomy'] = {'status': 'evaluated', 'method': 'tsxr_and_synthetic_landmark_heatmaps',
            'regions': regions, 'roi_proposals': [proposal] if proposal['status'] == 'proposed' else [],
            'neck_roi_assessment': proposal, 'landmark_points': points.tolist() if points is not None else None,
            'fixed_neck_roi': fixed_proposal, 'adapted_neck_roi': adapted_proposal,
            'own_roi_geometry': roi_geometry,
            'model_sha256': self.pelvis_sha256, 'model_hashes': self.hip_model_hashes,
            'landmark_model_used_sha256': landmark_sha256 or self.hip_model_hashes['landmarks'],
            'physical_measurements': extra, 'coordinate_units': 'original_image_pixels',
            'clinical_validation': False, 'affects_decision': self.hip_features_used if affects_decision is None else affects_decision,
            'target_semantics': 'projected mask and landmark/ROI proposals; not verified anatomical contours'}
        return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output', 'bank', 'ranking_bank', 'decoders', 'feature_spec', 'anatomy_model', 'baseline_profile'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--encoders', type=Path, nargs=2, required=True)
    parser.add_argument('--held-out-replay', action='store_true')
    parser.add_argument('--replay-manifest', type=Path)
    parser.add_argument('--pelvis-model', type=Path)
    parser.add_argument('--pelvis-checkpoint-sha256')
    parser.add_argument('--hip-landmarks', type=Path)
    parser.add_argument('--hip-projection', type=Path)
    parser.add_argument('--neck-template', type=Path)
    parser.add_argument('--disable-spine-visibility', action='store_true')
    parser.add_argument('--hip-display-landmarks', type=Path)
    parser.add_argument('--hip-display-template', type=Path)
    parser.add_argument('--require-network-isolation', action='store_true')
    args = parser.parse_args()
    if args.held_out_replay and args.replay_manifest is None:
        parser.error('--held-out-replay requires --replay-manifest')
    if bool(args.pelvis_model) != bool(args.pelvis_checkpoint_sha256):
        parser.error('Pelvic anatomy needs both --pelvis-model and --pelvis-checkpoint-sha256')
    network_interfaces = [name for _, name in socket.if_nameindex()]
    if args.require_network_isolation and set(network_interfaces) != {'lo'}:
        parser.error('--require-network-isolation needs a network namespace with loopback only')
    warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*')
    warnings.filterwarnings('ignore', message='TypedStorage is deprecated.*')
    os.environ['DXAQC_WORKFLOW_PROFILE'] = str(args.baseline_profile.resolve())
    candidate = LiveAnatomyCandidate(args.bank, args.ranking_bank, args.decoders, args.feature_spec,
                                    args.anatomy_model, args.encoders, held_out_replay=args.held_out_replay,
                                    replay_manifest=args.replay_manifest, pelvis_model=args.pelvis_model,
                                    pelvis_checkpoint_sha256=args.pelvis_checkpoint_sha256,
                                    hip_landmarks=args.hip_landmarks, hip_projection=args.hip_projection,
                                    neck_template=args.neck_template, visibility_cue=not args.disable_spine_visibility,
                                    hip_display_landmarks=args.hip_display_landmarks,
                                    hip_display_template=args.hip_display_template)
    summary = run_batch(args.input, args.output, Options(keep_explanation_dir=True), candidate,
                        progress=lambda i, n: print(f'{i}/{n}', flush=True) if i % 25 == 0 else None)
    (args.output / 'raw_candidate_predictions.json').write_text(json.dumps(candidate.records, indent=2) + '\n')
    (args.output / 'executed_live_anatomy_candidate.py').write_bytes(EXECUTION_SOURCE)
    (args.output / 'candidate_protocol.json').write_text(json.dumps({'profile': candidate.workflow_profile,
        'bank_sha256': candidate.manifest['bank_sha256'], 'code_sha256': EXECUTION_SHA256,
        'raw_feature_cache_used': False, 'target_labels_used_by_inference': False,
        'network_interfaces': network_interfaces,
        'replay_manifest_sha256': digest(args.replay_manifest) if args.replay_manifest else None,
        'pelvic_anatomy_sha256': candidate.pelvis_sha256,
        'clinical_validation': False, 'automatic_release_promotions': 0}, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
