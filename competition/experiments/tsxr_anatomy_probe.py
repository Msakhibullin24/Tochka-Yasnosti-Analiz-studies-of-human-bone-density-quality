"""Research-only transfer probe of the publisher's TSXR vertebral model.

Run in a separate nnunetv2ml environment, never the pinned release environment.
Weights: https://zenodo.org/records/17052912 (CC-BY-NC-4.0).
Presence/order counts are plausibility checks, not segmentation accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, default=ROOT / 'labels/image_labels.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--region', choices=('spine', 'hip', 'all'), default='spine')
    parser.add_argument('--horizontal-flip', action='store_true')
    parser.add_argument('--rotate-180', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    torch.set_num_threads(4)
    predictor = nnUNetPredictor(use_mirroring=False, allow_tqdm=False, verbose=False,
                               device=torch.device('cuda'))
    predictor.initialize_from_trained_model_folder(str(args.model), use_folds=(0,),
                                                   checkpoint_name='checkpoint_best.pth')
    args.output.mkdir(parents=True)
    labels = pd.read_csv(args.labels)
    region_filter = (labels.region.eq('spine') if args.region == 'spine' else
                     labels.region.str.startswith('hip') if args.region == 'hip' else labels.region.notna())
    labelled_filter = labels.quality_class.notna() if 'quality_class' in labels else np.ones(len(labels), dtype=bool)
    selected = labels.loc[region_filter & labelled_filter].head(args.limit)
    names = {int(value): name for name, value in predictor.dataset_json['labels'].items()}
    results, tiles = [], []
    for row in selected.itertuples():
        image = read_dxa(args.dataset / row.first_source_path)
        sx, sy = image.pixel_mm_x, image.pixel_mm
        for polarity in ('bone-bright', 'inverted'):
            pixels = image.pixels if polarity == 'bone-bright' else 255 - image.pixels
            if args.rotate_180:
                pixels = np.ascontiguousarray(pixels[::-1, ::-1])
            if args.horizontal_flip:
                pixels = np.ascontiguousarray(pixels[:, ::-1])
            started = perf_counter()
            segmentation = predictor.predict_single_npy_array(
                pixels[None, None].astype(np.float32), {'spacing': (999 * max(sx, sy), sy, sx)})
            elapsed = perf_counter() - started
            segmentation = np.asarray(segmentation).squeeze()
            multilabel = predictor.label_manager.multilabel
            expected = (len(names) - 1, *pixels.shape) if multilabel else pixels.shape
            if segmentation.shape != expected:
                raise ValueError(f'Unexpected output shape {segmentation.shape}')
            if args.horizontal_flip:
                segmentation = np.ascontiguousarray(segmentation[..., ::-1])
            if args.rotate_180:
                segmentation = np.ascontiguousarray(segmentation[..., ::-1, ::-1])
            centers, counts = {}, {}
            for value, name in names.items():
                if value == 0:
                    continue
                # nnunetv2ml calls independent overlapping channels 'multiclass'.
                mask = (segmentation[value - 1] if multilabel else segmentation == value).astype(np.uint8)
                n, cc, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
                if n <= 1:
                    continue
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                count = int(stats[largest, cv2.CC_STAT_AREA])
                if count < 4:
                    continue
                centers[name] = centroids[largest].tolist()
                counts[name] = count
            lumbar = [centers.get(f'vertebrae-l{i}') for i in range(1, 5)]
            complete = all(p is not None for p in lumbar)
            ordered = bool(complete and np.all(np.diff(np.array(lumbar)[:, 1]) > 0))
            angle = None
            if ordered:
                points = np.asarray(lumbar) * [sx, sy]
                angle = float(np.degrees(np.arctan(np.polyfit(points[:, 1], points[:, 0], 1)[0])))
            stem = f'{row.pixel_sha256[:16]}-{polarity}'
            np.save(args.output / f'{stem}.npy', segmentation)
            canvas = cv2.cvtColor(image.pixels, cv2.COLOR_GRAY2BGR)
            for name, center in centers.items():
                point = tuple(np.rint(center).astype(int))
                cv2.circle(canvas, point, 3, (0, 0, 255), -1)
                cv2.putText(canvas, name.replace('vertebrae-', ''), point,
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
            cv2.imwrite(str(args.output / f'{stem}.png'), canvas)
            if len(tiles) < 24:
                tiles.append(cv2.resize(canvas, (300, 300)))
            result = {'pixel_sha256': row.pixel_sha256, 'polarity': polarity,
                      'decoded_pixel_sha256': image.pixel_sha256,
                      'source_file_sha256': hashlib.file_digest((args.dataset / row.first_source_path).open('rb'), 'sha256').hexdigest(),
                      'mask_sha256': hashlib.file_digest((args.output / f'{stem}.npy').open('rb'), 'sha256').hexdigest(),
                      'seconds': elapsed, 'centers': centers, 'pixel_counts': counts,
                      'complete_l1_l4': complete, 'ordered_l1_l4': ordered,
                      'candidate_axis_degrees': angle, 'clinical_validation': False}
            results.append(result)
            print(json.dumps({k: result[k] for k in ('pixel_sha256', 'polarity', 'seconds',
                                                   'complete_l1_l4', 'ordered_l1_l4') }), flush=True)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(args.output / 'contact-sheet.png'),
                    np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]))
    checkpoint = args.model / 'fold_0/checkpoint_best.pth'
    report = {'source': 'https://zenodo.org/records/17052912', 'weight_license': 'CC-BY-NC-4.0',
              'labels_sha256': hashlib.file_digest(args.labels.open('rb'), 'sha256').hexdigest(),
              'code_sha256': hashlib.file_digest(Path(__file__).open('rb'), 'sha256').hexdigest(),
              'checkpoint_sha256': hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
              'images': len(selected), 'results': results, 'clinical_validation': False,
              'input_horizontally_flipped': bool(args.horizontal_flip),
              'input_rotated_180': bool(args.rotate_180),
              'input_has_qc_label_column': 'quality_class' in labels,
              'segmentation_accuracy_measured': False, 'automatic_promotions': 0,
              'limitation': 'No anatomical ground truth; counts/order do not prove correct numbering.'}
    (args.output / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
