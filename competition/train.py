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

from dxaqc.decision import decide, VERSION as DECISION_VERSION
from dxaqc.dicom_io import read_any, CALIBRATION_VERSION
from dxaqc.embedding import WEIGHTS_SHA256, embed
from dxaqc.geometry import measure_hip, measure_spine, measure_image
from dxaqc.model import CRITERIA, SEED, GroupModel, QualityBundle, RegionRouter, best_f1_threshold, group_of
from source_integrity import inspect_sources

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


def synthetic_spine(px: np.ndarray, kind: str, rng: np.random.Generator, angle_now: float, pixel_mm_y: float = 1.0,
                    pixel_mm_x: float = 1.0) -> np.ndarray:
    import cv2
    h, w = px.shape
    if kind == "rotate":
        target = rng.uniform(7.0, 12.0) * rng.choice([-1.0, 1.0])
        # cv2 rotates counter-clockwise for a positive angle, which measure_spine reports as +angle
        # (verified by tests/test_core.py::test_spine_axis_angle_is_measured_within_one_degree)
        # Rotate in physical coordinates, then map back to the original pixel grid.
        centre = ((w - 1) / 2, (h - 1) / 2)
        rotation = cv2.getRotationMatrix2D((0, 0), target - angle_now, 1.0)[:, :2]
        scale = np.diag([pixel_mm_x, pixel_mm_y])
        linear = np.linalg.solve(scale, rotation @ scale)
        m = np.column_stack([linear, np.asarray(centre) - linear @ centre])
        return cv2.warpAffine(px, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return np.ascontiguousarray(px[: int(h * rng.uniform(0.70, 0.78))])  # "crop": iliac crests cut off


def build_table(dataset: Path, labels: pd.DataFrame, cache: Path,
                source_fingerprint: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    sources = "".join((HERE / "dxaqc" / n).read_text() for n in ("geometry.py", "embedding.py", "dicom_io.py"))
    # The label CSV may be unchanged while source DICOM pixels have changed.
    # Bind cached measurements/embeddings to the audited decoded image set.
    key = hashlib.sha256((labels.to_csv() + sources + WEIGHTS_SHA256 +
                          str(dataset.resolve()) + source_fingerprint).encode()).hexdigest()[:16]
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
        feats.append(measure_image(px, r.region, img.pixel_mm, img.pixel_mm_x).features)
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


def pipeline_cv(dataset, labels, feats, emb, raw_emb, n_real, repeats):
    """Outer study folds shared by router and quality models; held-out raw DICOM inference."""
    from dxaqc.pipeline import Analyzer
    from dxaqc.model import official_violation_type
    y = labels.quality_class.values.astype(float)
    studies = labels.study_key.values
    reg = labels.region.values
    real = np.arange(n_real)
    output = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(cv.split(real, y[:n_real], studies[:n_real])):
            assert not set(studies[tr]) & set(studies[te])
            fitted = {}
            for group in ('spine', 'hip'):
                idx = np.array([i for i in tr if group_of(reg[i]) == group])
                targets = {k: pd.to_numeric(labels[k], errors='coerce').values.astype(float) for k in CRITERIA[group]}
                inner_s, inner_c = oof(group, feats, emb, y, targets, studies, idx, 4, SEED + 100 + rep)
                gm = fit_group(group, feats, emb, y, targets, idx)
                gm.quality_threshold = best_f1_threshold(y[idx].astype(int), inner_s)
                gm.criterion_thresholds = {}
                for k in targets:
                    known = ~np.isnan(targets[k][idx])
                    gm.criterion_thresholds[k] = best_f1_threshold(targets[k][idx][known].astype(int), inner_c[k][known])
                fitted[group] = gm
            route_idx = np.unique(np.concatenate([expand(tr, 'spine'), expand(tr, 'hip')]))
            router = RegionRouter().fit(raw_emb[route_idx], list(reg[route_idx]))
            analyzer = Analyzer(QualityBundle(router, fitted, {}))
            for i in te:
                img = read_any(dataset / labels.first_source_path[i])
                result = analyzer.analyze(img)
                output.append({'index': int(i), 'repeat': rep, 'fold': fold, 'study': str(studies[i]),
                               'source_path': labels.first_source_path[i], 'true_region': reg[i],
                               'predicted_region': result['region'], 'quality_true': int(y[i]),
                               'quality_score': result['score'], 'quality_pred': result['quality'],
                               'violation_type': official_violation_type(result['violations'])})
            print(f'pipeline CV repeat={rep} fold={fold} images={len(te)}', flush=True)
    table = pd.DataFrame(output)
    reports = []
    for rep in range(repeats):
        rows = table[table['repeat'] == rep]
        reports.append({'repeat': rep, 'quality': binary_metrics(rows.quality_true.values, rows.quality_score.values, rows.quality_pred.values),
                        'quality_ci95': cluster_bootstrap(rows.quality_true.values, rows.quality_score.values, rows.quality_pred.values, rows.study.values),
                        'router_accuracy': float((rows.true_region == rows.predicted_region).mean())})
    return table, reports


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
    ap.add_argument("--pipeline-cv", action="store_true", help="joint router/quality outer folds and raw input inference; writes report only")
    args = ap.parse_args()
    t0 = time.time()
    source_specs = [('organiser', args.labels, args.dataset)]
    for spec in args.extra:
        name, csv_path, root = spec.split('=', 2)
        source_specs.append((name, Path(csv_path), Path(root)))
    source_integrity = inspect_sources(source_specs)
    source_integrity_summary = {key: value for key, value in source_integrity.items() if key != 'entries'}
    source_integrity_summary['entries_sha256'] = hashlib.sha256(
        json.dumps(source_integrity['entries'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    source_fingerprints = {
        name: hashlib.sha256(json.dumps(
            [(entry['path'], entry['pixel_sha256_current']) for entry in source_integrity['entries']
             if entry['source'] == name], ensure_ascii=False).encode()).hexdigest()
        for name, _, _ in source_specs}
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    feats, emb, raw_emb = build_table(args.dataset, labels, HERE / ".cache",
                                      source_fingerprints['organiser'])
    n_real = len(labels)
    aux_parent: list[int] = []
    aux_source: dict[str, int] = {}
    for spec in args.extra:
        name, csv_path, root = spec.split("=", 2)
        extra = pd.read_csv(csv_path)
        extra = extra[extra.quality_class.notna() & extra.region.isin(["spine", "hip_right", "hip_left"])].reset_index(drop=True)
        extra["study_key"] = name + ":" + extra.study_key.astype(str)
        f2, e2, r2 = build_table(Path(root), extra, HERE / ".cache", source_fingerprints[name])
        feats, emb, raw_emb = feats + f2, np.vstack([emb, e2]), np.vstack([raw_emb, r2])
        labels = pd.concat([labels, extra], ignore_index=True)
        aux_parent += [-1] * len(extra)
        aux_source[name] = int(len(extra))
    if args.synthetic:
        from dxaqc.embedding import embed as _embed
        rng = np.random.default_rng(SEED)
        good = [i for i in range(n_real) if labels.region[i] == "spine" and labels.quality_class[i] == 0]
        syn_rows, syn_f, syn_e = [], [], []
        for i in good:
            img = read_any(args.dataset / labels.first_source_path[i])
            plan = [("rotate", "spine_axis")] * 2 if args.synthetic in ("rotate", "both") else []
            plan += [("crop", "spine_coverage")] if args.synthetic in ("crop", "both") else []
            for kind, crit in plan:
                px = synthetic_spine(img.pixels, kind, rng, feats[i]["spine_angle_deg"],
                                     img.pixel_mm, img.pixel_mm_x or img.pixel_mm)
                measurement = measure_image(px, "spine", img.pixel_mm, img.pixel_mm_x)
                if kind == "rotate" and not (np.isfinite(measurement.features["spine_abs_angle_deg"])
                                              and measurement.features["spine_abs_angle_deg"] > 5):
                    continue  # do not assign a positive angle label to a failed transform
                row = labels.iloc[i].copy()
                # The transform changes the whole frame (black corners, shifted anatomy), so the parent's other
                # criterion labels are no longer trustworthy: a synthetic sample teaches ONLY its target criterion.
                for other in CRITERIA["spine"]:
                    row[other] = np.nan
                row["quality_class"], row[crit] = 1, 1
                syn_rows.append(row), syn_f.append(measurement.features), syn_e.append(_embed(px))
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
    if args.pipeline_cv:
        table, results = pipeline_cv(args.dataset, labels, feats, emb, raw_emb, n_real, args.repeats)
        report = {'scope': 'study-held-out raw DICOM -> shared Analyzer with fold-trained router and quality models',
                  'limitations': 'binary quality and router metrics only; no independent clinical validation; not an evaluation of already-fitted shipped weights',
                  'calibration_version': CALIBRATION_VERSION, 'seed': SEED, 'synthetic': args.synthetic, 'auxiliary': aux_source,
                  'labels_sha256': hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                  'source_integrity': source_integrity_summary,
                  'training_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'repeats': results}
        directory = HERE / 'reports'
        directory.mkdir(exist_ok=True)
        (directory / f'{args.report_name}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        table.to_csv(directory / f'{args.report_name}_oof.csv', index=False)
        return
    report: dict = {"protocol": __doc__.split("Validation protocol")[1].strip(), "seed": SEED,
                    "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                    "encoder_sha256": WEIGHTS_SHA256, "images": int(n_real),
                    "studies": int(len(np.unique(studies[:n_real]))), "train_only_auxiliary_samples": aux_source,
                    "evaluation_scope": "organiser images only; auxiliary samples are never evaluated",
                    "decision_version": DECISION_VERSION,
                    "routing_scope": "quality evaluated with labelled region; router evaluated separately",
                    "aggregation": "mean score and majority decision over repeats; see per-repeat metrics",
                    "code_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in
                        sorted((HERE / "dxaqc").glob("*.py"))) + Path(__file__).read_bytes()).hexdigest(),
                    "source_integrity": source_integrity_summary,
                    "configuration": {"repeats": args.repeats, "synthetic": args.synthetic, "extra": args.extra}}
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
            crit_preds = {k: np.zeros((args.repeats, len(idx))) for k in CRITERIA[group]}
            for rep in range(args.repeats):
                cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep)
                for tr, te in cv.split(idx, quality[idx], studies[idx]):
                    inner_s, inner_c = oof(group, feats, emb, quality, criteria, studies, idx[tr], 4, SEED + 100 + rep)
                    thr = best_f1_threshold(quality[idx[tr]].astype(int), inner_s)
                    thresholds = {}
                    for k in CRITERIA[group]:
                        known_inner = ~np.isnan(criteria[k][idx[tr]])
                        thresholds[k] = best_f1_threshold(criteria[k][idx[tr]][known_inner].astype(int),
                                                          inner_c[k][known_inner])
                    m = fit_group(group, feats, emb, quality, criteria, idx[tr])
                    s, c = m.predict([feats[i] for i in idx[te]], emb[idx[te]])
                    for j, pos in enumerate(te):
                        decision = decide(group, float(s[j]), {k: float(c[k][j]) for k in c},
                                          feats[idx[pos]], thr, thresholds)
                        scores[rep, pos], preds[rep, pos] = decision["score"], decision["quality"]
                        for k in c:
                            crit_scores[k][rep, pos] = c[k][j]
                            # Implant and ROI share the organiser's official violation label.
                            crit_preds[k][rep, pos] = k in decision["violations"] or (
                                k == "hip_roi_coverage" and "hip_metal_implant" in decision["violations"])

                print(f"[{group}] repeat {rep} AUC={roc_auc_score(quality[idx], scores[rep]):.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            y = quality[idx].astype(int)
            mean_s = scores.mean(0)
            vote = (preds.mean(0) >= 0.5).astype(int)
            final_thr = best_f1_threshold(y, mean_s)
            g = {"quality": binary_metrics(y, mean_s, vote),
                 "quality_ci95": cluster_bootstrap(y, mean_s, vote, studies[idx]),
                 "quality_auc_per_repeat": [float(roc_auc_score(y, r)) for r in scores],
                 "quality_per_repeat": [binary_metrics(y, scores[r], preds[r]) for r in range(args.repeats)],
                 "criteria": {}}
            f1s = []
            for k in CRITERIA[group]:
                known = ~np.isnan(criteria[k][idx])
                yk, sk = criteria[k][idx][known].astype(int), crit_scores[k].mean(0)[known]
                crit_thr[k] = best_f1_threshold(yk, sk)
                pk = (crit_preds[k].mean(0)[known] >= 0.5).astype(int)
                mk = binary_metrics(yk, sk, pk)
                mk["ci95"] = cluster_bootstrap(yk, sk, pk, studies[idx][known], 1000)
                mk["note"] = "inner-fold thresholds; final decision rules applied; majority vote over repeats"
                mk["per_repeat"] = [binary_metrics(yk, crit_scores[k][r][known], crit_preds[k][r][known])
                                    for r in range(args.repeats)]
                oof_rows.loc[idx, k + "_score"] = crit_scores[k].mean(0)
                oof_rows.loc[idx, k + "_pred"] = (crit_preds[k].mean(0) >= 0.5).astype(int)
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
        oof_name = "oof_predictions.csv" if args.report_name == "validation_metrics" else args.report_name + "_oof.csv"
        oof_rows.to_csv(HERE / "reports" / oof_name, index=False)
        print(json.dumps(report["overall"], indent=1))
    import sklearn
    bundle = QualityBundle(router, groups, {"trained_at_unix": int(t0), "calibration_version": CALIBRATION_VERSION, "sklearn": sklearn.__version__,
                                            "labels_sha256": report["labels_sha256"], "seed": SEED,
                                            "code_sha256": report["code_sha256"], "decision_version": DECISION_VERSION,
                                            "configuration": report["configuration"]})
    if args.no_save_model:
        print("experiment finished in", round(time.time() - t0), "s (model not saved)")
        return
    (HERE / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, HERE / "models" / "bundle.joblib", compress=3)
    print("saved models/bundle.joblib in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
