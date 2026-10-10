"""Is the new corpus the same kind of object as the competition frames?

Self-supervised pretraining on fourteen thousand DXA images is the one remaining route that
could beat a 249-image ceiling, so before spending GPU days on it the cheap question is asked
first: are these the same kind of picture as the competition frames at all?

Three checks, in increasing order of how much they would hurt if they failed. Shape and bit
depth, because the competition frames are small fixed-size greyscale DXA frames and a corpus of
cropped region-of-interest thumbnails would defeat every feature extractor in the catalogue,
whose letterbox pipeline assumes a complete frame. Intensity distribution, because the
competition frames share one device calibration and a differently sourced corpus will not.
And pixel-hash overlap against the 249 labelled frames, which different institutions make very
unlikely but which is checked rather than assumed.

The result decides whether a pretraining run is worth starting at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gpu_research.common import dump  # noqa: E402
from source_integrity import inspect_sources  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def describe(a: np.ndarray) -> dict:
    return {'shape': [int(a.shape[0]), int(a.shape[1])], 'min': int(a.min()), 'max': int(a.max()),
            'mean': float(a.mean()), 'std': float(a.std()),
            'p01': float(np.percentile(a, 1)), 'p99': float(np.percentile(a, 99)),
            'frac_zero': float((a == 0).mean()), 'frac_max': float((a == a.max()).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--competition-dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    parser.add_argument('--sample', type=int, default=400)
    parser.add_argument('--hash-sample', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)

    inv = pd.read_csv(args.inventory)
    scope = inv[inv.in_qa_scope].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    pick = scope.iloc[rng.permutation(len(scope))[:args.sample]]

    rows = []
    for r in pick.itertuples():
        try:
            im = Image.open(args.root / r.path)
            a = np.asarray(im.convert('L'))
        except Exception as exc:  # noqa: BLE001
            rows.append({'path': r.path, 'error': f'{type(exc).__name__}: {exc}'}); continue
        mode = im.mode
        rec = {'path': r.path, 'folder': r.folder, 'patient': r.patient, 'pil_mode': mode,
               'file_mode_dtype': str(im.info.get('dpi'))}
        rec.update(describe(a))
        rows.append(rec)
    new = pd.DataFrame(rows)
    ok = new[new.get('error').isna() if 'error' in new.columns else slice(None)]
    new.to_csv(args.output / 'corpus_sample_stats.csv', index=False)

    comp = []
    # Read the competition frames through the project's own loader so the uint8 path and
    # polarity handling are exactly the ones the models see.
    from dxaqc.dicom_io import read_dxa
    labels = pd.read_csv(args.labels)
    labels = labels[labels.quality_class.notna()].reset_index(drop=True)
    for r in labels.itertuples():
        p = args.competition_dataset / r.first_source_path
        try:
            a = read_dxa(p).pixels
        except Exception as exc:  # noqa: BLE001
            comp.append({'error': str(exc)}); continue
        rec = {'path': r.first_source_path, 'region': r.region, 'rows': int(r.rows), 'columns': int(r.columns)}
        rec.update(describe(a))
        comp.append(rec)
    compd = pd.DataFrame([c for c in comp if 'error' not in c])
    compd.to_csv(args.output / 'competition_sample_stats.csv', index=False)

    def agg(df, key):
        g = df.groupby(key).agg(
            n=('shape', 'size'),
            height_min=('shape', lambda s: int(min(x[0] for x in s))),
            height_max=('shape', lambda s: int(max(x[0] for x in s))),
            width_min=('shape', lambda s: int(min(x[1] for x in s))),
            width_max=('shape', lambda s: int(max(x[1] for x in s))),
            mean_mean=('mean', 'mean'), mean_std=('std', 'mean'),
            mean_frac_zero=('frac_zero', 'mean')).round(4)
        return g.to_dict(orient='index')

    # Pixel-hash overlap, both directions.
    comp_hashes = set(labels.pixel_sha256.dropna().astype(str))
    hash_pick = scope.iloc[rng.permutation(len(scope))[:args.hash_sample]]
    new_hashes, new_paths = set(), {}
    for r in hash_pick.itertuples():
        try:
            a = np.asarray(Image.open(args.root / r.path).convert('L'))
        except Exception:  # noqa: BLE001
            continue
        h = hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()
        new_hashes.add(h); new_paths[h] = r.path

    aspect_new = ok['shape'].apply(lambda s: s[0] / s[1])
    aspect_comp = compd['shape'].apply(lambda s: s[0] / s[1])
    report = {
        'protocol': __doc__,
        'corpus': {'sampled': int(len(ok)), 'by_folder': agg(ok, 'folder'),
                   'aspect_ratio': {'median': float(aspect_new.median()),
                                    'min': float(aspect_new.min()), 'max': float(aspect_new.max())},
                   'pil_modes': {k: int(v) for k, v in ok.pil_mode.value_counts().items()},
                   'read_errors': int(new.error.notna().sum()) if 'error' in new.columns else 0},
        'competition': {'sampled': int(len(compd)), 'by_region': agg(compd, 'region'),
                        'aspect_ratio': {'median': float(aspect_comp.median()),
                                         'min': float(aspect_comp.min()), 'max': float(aspect_comp.max())}},
        'overlap': {'competition_hashes': int(len(comp_hashes)),
                    'corpus_hashes_checked': int(len(new_hashes)),
                    'shared_hashes': len(comp_hashes & new_hashes),
                    'collisions': [{'sha256': h, 'corpus_path': new_paths[h]} for h in (comp_hashes & new_hashes)],
                    'note': 'Exact pixel-identity only. Near-duplicates are not detected here and '
                            'would need perceptual hashing, but two institutions should not produce them.'},
        'clinical_validation': False, 'model_affects_decision': False}
    dump(args.output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('corpus', 'competition', 'overlap')}, indent=2, default=str)[:4000])


if __name__ == '__main__':
    main()
