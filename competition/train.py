"""Train and honestly validate the DXA quality bundle.

    python train.py --dataset "/path/НД_для_обучения/Исследования" [--repeats 3] [--skip-cv]
                    [--extra NAME=labels.csv=/path/to/root ...] [--synthetic rotate|crop|both]

--extra      additional labelled datasets (same CSV schema as labels/image_labels.csv; DICOM or PNG/JPG).
             They are used for TRAINING ONLY. All reported metrics stay organiser-only, so runs with and
             without extra data are directly comparable and extra data can never inflate the numbers.
--synthetic  label-preserving synthetic violations made from good organiser spine images (rotation beyond
             the 5 degree limit, cropping away the iliac crests). Train-only, children follow their parent
             image into the training fold; evaluation is always on real images.

Inputs : reviewed per-image labels (labels/image_labels.csv) + raw organiser DICOM folders.
Outputs: models/bundle.joblib, reports/validation_metrics.json, reports/oof_predictions.csv

Validation protocol
  * unit of splitting = study folder (all images / exact pixel copies of a study stay together);
  * repeated StratifiedGroupKFold(5); decision thresholds are chosen on INNER out-of-fold
    predictions of the training part only, so reported F1 is not tuned on the evaluated fold;
  * 95% CIs: bootstrap over studies (cluster bootstrap), 2000 resamples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from dxaqc.dicom_io import read_any
from dxaqc.embedding import WEIGHTS_SHA256, embed
from dxaqc.geometry import measure_hip, measure_spine
from dxaqc.model import CRITERIA, SEED, GroupModel, QualityBundle, RegionRouter, best_f1_threshold, group_of

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


# Train-only auxiliary samples (extra datasets, synthetic violations). parent = index of the real organiser
# image a synthetic sample was made from, -1 for an independent extra-dataset image.
AUX = {"idx": np.array([], int), "parent": np.array([], int), "group": np.array([], object)}


def expand(train_idx: np.ndarray, group: str) -> np.ndarray:
    """Training indices = real training images + auxiliary samples that cannot leak the evaluated fold."""
    if AUX["idx"].size == 0:
        return train_idx
    ok = (AUX["group"] == group) & ((AUX["parent"] < 0) | np.isin(AUX["parent"], train_idx))
    return np.concatenate([train_idx, AUX["idx"][ok]])


def synthetic_spine(px: np.ndarray, kind: str, rng: np.random.Generator, angle_now: float) -> np.ndarray:
    import cv2
    h, w = px.shape
    if kind == "rotate":
        target = rng.uniform(7.0, 12.0) * rng.choice([-1.0, 1.0])
        # cv2 rotates counter-clockwise for a positive angle, which measure_spine reports as +angle
        # (verified by tests/test_core.py::test_spine_axis_angle_is_measured_within_one_degree)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), target - angle_now, 1.0)
        return cv2.warpAffine(px, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return np.ascontiguousarray(px[: int(h * rng.uniform(0.70, 0.78))])  # "crop": iliac crests cut off


def build_table(dataset: Path, labels: pd.DataFrame, cache: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    sources = "".join((HERE / "dxaqc" / n).read_text() for n in ("geometry.py", "embedding.py", "dicom_io.py"))
    key = hashlib.sha256((labels.to_csv() + sources + WEIGHTS_SHA256 + str(dataset.resolve())).encode()).hexdigest()[:16]
    f = cache / f"train_table_{key}.joblib"
    if f.exists():
        return joblib.load(f)
    feats, embs, raw_embs = [], [], []
    for r in labels.itertuples():
        img = read_any(dataset / r.first_source_path, getattr(r, "pixel_mm", None))
        px = img.pixels
        raw_embs.append(embed(px))  # un-mirrored: the router must see the true side
        if r.region == "hip_left":
            px = np.ascontiguousarray(px[:, ::-1])
        feats.append((measure_spine if r.region == "spine" else measure_hip)(px, img.pixel_mm).features)
        embs.append(embed(px))
    out = (feats, np.stack(embs), np.stack(raw_embs))
    cache.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, f)
    return out


def binary_metrics(y: np.ndarray, s: np.ndarray, pred: np.ndarray) -> dict:
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    out = {"n": int(len(y)), "positives": int(y.sum()), "sensitivity": sens, "specificity": spec,
           "balanced_accuracy": (sens + spec) / 2, "f1": 2 * tp / max(2 * tp + fp + fn, 1),
           "confusion_tn_fp_fn_tp": [tn, fp, fn, tp]}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    return out


def cluster_bootstrap(y, s, pred, groups, n=2000) -> dict:
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    acc: dict[str, list] = {}
    for _ in range(n):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq))])
        m = binary_metrics(y[idx], s[idx], pred[idx])
        for k in ("sensitivity", "specificity", "balanced_accuracy", "f1", "roc_auc", "pr_auc"):
            if k in m:
                acc.setdefault(k, []).append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}


def fit_group(group, feats, emb, quality, criteria, idx) -> GroupModel:
    idx = expand(np.asarray(idx), group)
    return GroupModel(group).fit([feats[i] for i in idx], emb[idx], quality[idx], {k: v[idx] for k, v in criteria.items()})


def oof(group, feats, emb, quality, criteria, studies, idx, splits, seed):
    """Out-of-fold quality score + criterion scores for the samples in idx."""
    score, crit = np.zeros(len(idx)), {k: np.zeros(len(idx)) for k in CRITERIA[group]}
    cv = StratifiedGroupKFold(splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(idx, quality[idx], studies[idx]):
        m = fit_group(group, feats, emb, quality, criteria, idx[tr])
        s, c = m.predict([feats[i] for i in idx[te]], emb[idx[te]])
        score[te] = s
        for k in c:
            crit[k][te] = c[k]
    return score, crit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--labels", type=Path, default=HERE / "labels" / "image_labels.csv")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=CSV=ROOT")
    ap.add_argument("--synthetic", choices=["rotate", "crop", "both"], default=None)
    ap.add_argument("--report-name", default="validation_metrics", help="file stem in reports/ (keep runs side by side)")
    ap.add_argument("--no-save-model", action="store_true", help="experiment only: do not overwrite models/bundle.joblib")
    args = ap.parse_args()
    t0 = time.time()
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    feats, emb, raw_emb = build_table(args.dataset, labels, HERE / ".cache")
    n_real = len(labels)
    aux_parent: list[int] = []
    aux_source: dict[str, int] = {}
    for spec in args.extra:
        name, csv_path, root = spec.split("=", 2)
        extra = pd.read_csv(csv_path)
        extra = extra[extra.quality_class.notna() & extra.region.isin(["spine", "hip_right", "hip_left"])].reset_index(drop=True)
        extra["study_key"] = name + ":" + extra.study_key.astype(str)
        f2, e2, r2 = build_table(Path(root), extra, HERE / ".cache")
        feats, emb, raw_emb = feats + f2, np.vstack([emb, e2]), np.vstack([raw_emb, r2])
        labels = pd.concat([labels, extra], ignore_index=True)
        aux_parent += [-1] * len(extra)
        aux_source[name] = int(len(extra))
    if args.synthetic:
        from dxaqc.embedding import embed as _embed
        from dxaqc.geometry import measure_spine as _ms
        rng = np.random.default_rng(SEED)
        good = [i for i in range(n_real) if labels.region[i] == "spine" and labels.quality_class[i] == 0]
        syn_rows, syn_f, syn_e = [], [], []
        for i in good:
            img = read_any(args.dataset / labels.first_source_path[i])
            plan = [("rotate", "spine_axis")] * 2 if args.synthetic in ("rotate", "both") else []
            plan += [("crop", "spine_coverage")] if args.synthetic in ("crop", "both") else []
            for kind, crit in plan:
                px = synthetic_spine(img.pixels, kind, rng, feats[i]["spine_angle_deg"])
                row = labels.iloc[i].copy()
                # The transform changes the whole frame (black corners, shifted anatomy), so the parent's other
                # criterion labels are no longer trustworthy: a synthetic sample teaches ONLY its target criterion.
                for other in CRITERIA["spine"]:
                    row[other] = np.nan
                row["quality_class"], row[crit] = 1, 1
                syn_rows.append(row), syn_f.append(_ms(px, img.pixel_mm).features), syn_e.append(_embed(px))
                aux_parent.append(i)
        labels = pd.concat([labels, pd.DataFrame(syn_rows)], ignore_index=True)
        feats, emb = feats + syn_f, np.vstack([emb, np.stack(syn_e)])
        raw_emb = np.vstack([raw_emb, np.stack(syn_e)])
        aux_source["synthetic_spine"] = len(syn_rows)
    AUX["idx"] = np.arange(n_real, len(labels))
    AUX["parent"] = np.array(aux_parent, int)
    AUX["group"] = np.array([group_of(r) for r in labels.region.values[n_real:]], object)
    studies = labels.study_key.values
    quality = labels.quality_class.values.astype(float)
    report: dict = {"protocol": __doc__.split("Validation protocol")[1].strip(), "seed": SEED,
                    "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                    "encoder_sha256": WEIGHTS_SHA256, "images": int(n_real),
                    "studies": int(len(np.unique(studies[:n_real]))), "train_only_auxiliary_samples": aux_source,
                    "evaluation_scope": "organiser images only; auxiliary samples are never evaluated"}
    oof_rows = labels.iloc[:n_real][["pixel_sha256", "study_key", "region", "quality_class"]].copy()
    oof_rows["oof_score"], oof_rows["oof_pred"] = np.nan, np.nan

    # ---------------- region router ----------------
    reg = labels.region.values
    if not args.skip_cv:
        hits = np.zeros(n_real, bool)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(raw_emb[:n_real], reg[:n_real], studies[:n_real]):
            pred, _ = RegionRouter().fit(raw_emb[tr], list(reg[tr])).predict(raw_emb[te])
            hits[te] = np.array(pred) == reg[te]
        report["region_router_cv_accuracy"] = float(hits.mean())
        print("region router CV accuracy:", hits.mean(), flush=True)
    router = RegionRouter().fit(raw_emb, list(reg))

    # ---------------- quality models ----------------
    groups: dict[str, GroupModel] = {}
    for group in ("spine", "hip"):
        idx = np.where(np.array([group_of(r) for r in reg[:n_real]]) == group)[0]  # real organiser images only
        criteria = {k: pd.to_numeric(labels[k], errors="coerce").values.astype(float) for k in CRITERIA[group]}
        final_thr, crit_thr = 0.5, {k: 0.5 for k in CRITERIA[group]}
        if not args.skip_cv:
            scores = np.zeros((args.repeats, len(idx)))
            preds = np.zeros((args.repeats, len(idx)))
            crit_scores = {k: np.zeros((args.repeats, len(idx))) for k in CRITERIA[group]}
            for rep in range(args.repeats):
                cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep)
                for tr, te in cv.split(idx, quality[idx], studies[idx]):
                    inner_s, _ = oof(group, feats, emb, quality, criteria, studies, idx[tr], 4, SEED + 100 + rep)
                    thr = best_f1_threshold(quality[idx[tr]].astype(int), inner_s)
                    m = fit_group(group, feats, emb, quality, criteria, idx[tr])
                    s, c = m.predict([feats[i] for i in idx[te]], emb[idx[te]])
                    scores[rep, te], preds[rep, te] = s, s >= thr
                    for k in c:
                        crit_scores[k][rep, te] = c[k]
                print(f"[{group}] repeat {rep} AUC={roc_auc_score(quality[idx], scores[rep]):.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            y = quality[idx].astype(int)
            mean_s = scores.mean(0)
            vote = (preds.mean(0) >= 0.5).astype(int)
            final_thr = best_f1_threshold(y, mean_s)
            g = {"quality": binary_metrics(y, mean_s, vote),
                 "quality_ci95": cluster_bootstrap(y, mean_s, vote, studies[idx]),
                 "quality_auc_per_repeat": [float(roc_auc_score(y, r)) for r in scores],
                 "criteria": {}}
            f1s = []
            for k in CRITERIA[group]:
                known = ~np.isnan(criteria[k][idx])
                yk, sk = criteria[k][idx][known].astype(int), crit_scores[k].mean(0)[known]
                crit_thr[k] = best_f1_threshold(yk, sk)
                mk = binary_metrics(yk, sk, (sk >= crit_thr[k]).astype(int))
                mk["ci95"] = cluster_bootstrap(yk, sk, (sk >= crit_thr[k]).astype(int), studies[idx][known], 1000)
                mk["note"] = "criterion threshold chosen on pooled OOF (optimistic F1); AUC is threshold-free"
                g["criteria"][k] = mk
                f1s.append(mk["f1"])
            g["violation_macro_f1"] = float(np.mean(f1s))
            report[group] = g
            oof_rows.loc[idx, "oof_score"], oof_rows.loc[idx, "oof_pred"] = mean_s, vote
            print(json.dumps({group: g["quality"], "ci": g["quality_ci95"]}, indent=1), flush=True)
        model = fit_group(group, feats, emb, quality, criteria, idx)
        model.quality_threshold, model.criterion_thresholds = final_thr, crit_thr
        groups[group] = model

    if not args.skip_cv:
        ok = oof_rows.oof_score.notna().values
        y, s, p = quality[:n_real][ok].astype(int), oof_rows.oof_score.values[ok], oof_rows.oof_pred.values[ok].astype(int)
        report["overall"] = {"quality": binary_metrics(y, s, p),
                             "quality_ci95": cluster_bootstrap(y, s, p, studies[:n_real][ok])}
        (HERE / "reports").mkdir(exist_ok=True)
        (HERE / "reports" / f"{args.report_name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        if args.report_name == "validation_metrics":
            oof_rows.to_csv(HERE / "reports" / "oof_predictions.csv", index=False)
        print(json.dumps(report["overall"], indent=1))
    import sklearn
    bundle = QualityBundle(router, groups, {"trained_at_unix": int(t0), "sklearn": sklearn.__version__,
                                            "labels_sha256": report["labels_sha256"], "seed": SEED})
    if args.no_save_model:
        print("experiment finished in", round(time.time() - t0), "s (model not saved)")
        return
    (HERE / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, HERE / "models" / "bundle.joblib", compress=3)
    print("saved models/bundle.joblib in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
