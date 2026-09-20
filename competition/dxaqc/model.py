"""Quality model: region router + per-criterion ensembles + quality composition.

Learned scores are shared by training and inference. Final decisions live in decision.py.
Validation evaluates the region router separately from region-conditional quality models.

Per criterion (e.g. spine_axis) three weak learners are averaged:
  * RandomForest on a compact, criterion-specific set of geometric measurements (mm / degrees);
  * LogisticRegression on the same measurements;
  * LogisticRegression on the frozen ResNet18 embedding.
quality score = mean( noisy-OR over criteria , direct RF on all measurements , direct CNN-LR ).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REGIONS = ("spine", "hip_right", "hip_left")
# Output vocabulary fixed by the organiser (curator's answers, 18.09.2026): closed lists, exact spelling,
# several violations joined with ";", empty string when there is none; the hip side does not matter.
REGION_LABEL = {
    "spine": "Поясничный отдел позвоночника",
    "hip_right": "Проксимальный отдел бедра",
    "hip_left": "Проксимальный отдел бедра",
}
VIOLATION_LABEL = {
    "spine_coverage": "Некорректная укладка",
    "spine_axis": "Не выравнена ось позвоночника",
    "spine_artifact": "Присутствуют посторонние предметы",
    "hip_position_rotation": "Некорректная укладка",
    "hip_roi_coverage": "Некорректная область интереса",
    # Not in the organiser list: a prosthesis makes correct ROI placement impossible -> closest official label.
    "hip_metal_implant": "Некорректная область интереса",
}
LATERALITY = {"spine": "", "hip_right": "right", "hip_left": "left"}


def official_violation_type(codes: list[str]) -> str:
    """Internal codes -> organiser labels, de-duplicated, order preserved, ';' separated."""
    seen: list[str] = []
    for c in codes:
        label = VIOLATION_LABEL.get(c)
        if label and label not in seen:
            seen.append(label)
    return ";".join(seen)


CRITERIA = {
    "spine": {
        "spine_coverage": ["crest_left_frac", "crest_right_frac", "crest_min_frac", "crest_height_mm",
                           "image_height_mm", "rib_signal"],
        "spine_axis": ["spine_abs_angle_deg", "spine_seg_max_abs_angle_deg", "spine_curve_rms_mm",
                       "spine_curve_max_mm", "spine_abs_centre_offset_mm", "spine_seg_angle_range_deg"],
        "spine_artifact": ["tophat_lat_p999", "tophat_lat_p99", "tophat_lat_strong_frac", "tophat_col_p999",
                           "tophat_top_lat_p999", "tophat_top_lat_p99", "tophat_top_strong_frac",
                           "sat_pixels", "sat_max_blob", "rib_signal", "lateral_mean"],
    },
    "hip": {
        "hip_position_rotation": ["shaft_abs_angle_deg", "shaft_angle_deg", "lesser_troch_protrusion_mm",
                                  "lesser_troch_area_mm2", "lesser_troch_notch_mm", "medial_concavity_mm",
                                  "neck_start_height_mm", "shaft_width_mm", "lateral_margin_mm",
                                  "troch_top_margin_mm", "bone_frac"],
        "hip_roi_coverage": ["image_height_mm", "ischium_bottom_margin_mm", "troch_top_margin_mm",
                             "lateral_margin_mm", "bone_top_rows_frac", "ischium_touches_bottom"],
    },
}
VIOLATION_RU = {
    "spine_coverage": "Некорректная укладка: не визуализированы верхние края подвздошных костей и/или половина Th12",
    "spine_axis": "Ось позвоночника не выровнена (наклон более 5°)",
    "spine_artifact": "Посторонние предметы / артефакты / наложения в зоне исследования",
    "hip_position_rotation": "Некорректное позиционирование или ротация бедра (оценка малого вертела, оси диафиза)",
    "hip_roi_coverage": "Некорректная область интереса: недостаточные отступы (3 см сверху/снизу, 2 см сбоку)",
    "hip_metal_implant": "Металлоконструкция / эндопротез в зоне исследования",
}
SEED = 17


def group_of(region: str) -> str:
    return "spine" if region == "spine" else "hip"


def _rf():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                               random_state=SEED, n_jobs=1),
    )


def _lr(c: float = 0.3):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=SEED))


def _cnn_lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000, random_state=SEED))


def _matrix(feature_dicts: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[d.get(n, np.nan) for n in names] for d in feature_dicts], dtype=np.float64)


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


@dataclass
class GroupModel:
    """Models for one anatomical group ('spine' or 'hip')."""
    group: str
    all_features: list[str] = field(default_factory=list)
    criterion_models: dict = field(default_factory=dict)  # name -> (rf, lr, cnn) or None if untrainable
    direct_rf: object = None
    direct_cnn: object = None
    quality_threshold: float = 0.5
    criterion_thresholds: dict = field(default_factory=dict)

    def fit(self, feats: list[dict], emb: np.ndarray, quality: np.ndarray, criteria: dict[str, np.ndarray]) -> "GroupModel":
        self.all_features = sorted({k for d in feats for k in d})
        for name, cols in CRITERIA[self.group].items():
            y = criteria[name]
            known = ~np.isnan(y)
            if known.sum() < 10 or len(np.unique(y[known])) < 2:
                self.criterion_models[name] = None
                continue
            X = _matrix(feats, cols)[known]
            yy = y[known].astype(int)
            self.criterion_models[name] = (_rf().fit(X, yy), _lr().fit(X, yy), _cnn_lr().fit(emb[known], yy))
        Xa = _matrix(feats, self.all_features)
        q = quality.astype(int)
        self.direct_rf = _rf().fit(Xa, q)
        self.direct_cnn = _cnn_lr().fit(emb, q)
        return self

    def predict(self, feats: list[dict], emb: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        crit: dict[str, np.ndarray] = {}
        for name, cols in CRITERIA[self.group].items():
            models = self.criterion_models.get(name)
            if models is None:
                crit[name] = np.zeros(len(feats))
                continue
            X = _matrix(feats, cols)
            crit[name] = (_proba(models[0], X) + _proba(models[1], X) + _proba(models[2], emb)) / 3.0
        noisy_or = 1.0 - np.prod([1.0 - p for p in crit.values()], axis=0)
        Xa = _matrix(feats, self.all_features)
        score = (noisy_or + _proba(self.direct_rf, Xa) + _proba(self.direct_cnn, emb)) / 3.0
        return score, crit


@dataclass
class RegionRouter:
    model: object = None

    def fit(self, emb: np.ndarray, regions: list[str]) -> "RegionRouter":
        y = np.array([REGIONS.index(r) for r in regions])
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, random_state=SEED))
        self.model.fit(emb, y)
        return self

    def predict(self, emb: np.ndarray) -> tuple[list[str], np.ndarray]:
        p = self.model.predict_proba(emb)
        return [REGIONS[i] for i in p.argmax(1)], p.max(1)


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold maximising F1; ties resolved toward the midpoint between neighbouring scores."""
    order = np.unique(s)
    if not len(order):
        return 1.0
    if y.sum() == 0:
        return float(np.nextafter(order[-1], np.inf))
    cands = np.r_[order[0], (order[:-1] + order[1:]) / 2]
    best_t, best_f = 0.5, -1.0
    for t in cands:
        pred = s >= t
        tp = float((pred & (y == 1)).sum())
        f = 2 * tp / max(pred.sum() + y.sum(), 1)
        if f > best_f + 1e-12:
            best_f, best_t = f, float(t)
    return best_t


@dataclass
class QualityBundle:
    router: RegionRouter
    groups: dict[str, GroupModel]
    meta: dict = field(default_factory=dict)
