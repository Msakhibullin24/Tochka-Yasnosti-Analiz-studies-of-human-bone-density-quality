"""Convert the LUMOS archive into a compact, labelled training array.

LUMOS is 803 patients / 4838 lumbar radiographs with DXA-measured L1-L4 BMD and
T-scores. It is the only large ungated corpus with real bone-density ground truth
available without an application, so it is used here purely as a *pretext* source:
train an encoder to regress DXA-measured density from the image, then reuse its
features on the organizer DXA frames.

Nothing about the organiser labels is touched by this script.

Reads the DICOMs directly from the zip archives, keeps one standardised
view per patient-region so the pretext task is not dominated by repeats, and
writes uint8 arrays plus the label table.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SIZE = 256


def to_uint8(pixels):
    """Min-max normalise to uint8 so every exposure is comparable."""
    p = pixels.astype(np.float32)
    lo, hi = float(np.percentile(p, 1)), float(np.percentile(p, 99))
    if hi <= lo:
        return np.zeros(p.shape, np.uint8)
    return np.clip((p - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def load_dicom(blob):
    import pydicom
    from pydicom.pixels import pixel_array
    ds = pydicom.dcmread(io.BytesIO(blob), force=True)
    array = pixel_array(ds)
    if array.ndim == 3:
        array = array[0] if array.shape[0] <= 4 else array.mean(axis=0)
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-per-patient-region', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    args.output.mkdir(parents=True)

    clinical = pd.read_excel(args.root / 'lumos_clinical_data.xlsx')
    label_column = {'patient_id': 'patient_id', 'bmd': 'bmd_L1-L4', 'tscore': 'T_L1-L4',
                    'age': 'age', 'gender': 'gender', 'bmi': 'BMI'}
    have = {k: v for k, v in label_column.items() if v in clinical.columns}
    clinical = clinical.dropna(subset=[have['bmd'], have['tscore']])
    clinical = clinical.set_index(have['patient_id'])

    images, labels, source = [], [], []
    seen = set()
    skipped = {'read': 0, 'duplicate': 0, 'unmatched_patient': 0}
    pattern = re.compile(r'lumos_x_(\d+)')
    for archive in sorted(args.root.glob('lumos_x_*.zip')):
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist()
                     if n.lower().endswith('.dcm') and '__MACOSX' not in n]
            for name in names:
                stem = Path(name).stem
                match = pattern.match(Path(name).parts[0] if len(Path(name).parts) > 1 else stem)
                if not match:
                    continue
                number = int(match.group(1))
                if number not in clinical.index:
                    skipped['unmatched_patient'] += 1
                    continue
                # lumos_x_<n>_<region>_<view>.<ext>: keep one view per region.
                region = re.sub(r'^.*?_\d+_', '', stem)
                region = re.sub(r'_\d+$', '', region)
                key = (number, region)
                if key in seen:
                    skipped['duplicate'] += 1
                    continue
                try:
                    pixels = to_uint8(load_dicom(zf.read(name)))
                except Exception:
                    skipped['read'] += 1
                    continue
                if pixels.ndim != 2 or min(pixels.shape) < 32:
                    skipped['read'] += 1
                    continue
                import cv2
                h, w = pixels.shape
                scale = SIZE / max(h, w)
                resized = cv2.resize(pixels, (max(1, round(w * scale)), max(1, round(h * scale))))
                canvas = np.zeros((SIZE, SIZE), np.uint8)
                y, x = (SIZE - resized.shape[0]) // 2, (SIZE - resized.shape[1]) // 2
                canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                seen.add(key)
                images.append(canvas)
                row = clinical.loc[number]
                labels.append([float(row[have['bmd']]), float(row[have['tscore']]),
                               float(row[have['age']]) if have.get('age') else np.nan,
                               float(row[have['bmi']]) if have.get('bmi') else np.nan,
                               1.0 if str(row.get(have.get('gender', ''), '')).lower().startswith('f') else 0.0])
                source.append({'patient_id': int(number), 'region': region, 'file': name})
        print(f'{archive.name}: cumulative {len(images)} images', flush=True)

    if len(images) < 200:
        raise RuntimeError(f'Only {len(images)} usable images; refusing to train a pretext task on that')
    stack = np.stack(images)
    target = np.asarray(labels, np.float32)
    np.save(args.output / 'images.npy', stack)
    np.save(args.output / 'labels.npy', target)
    pd.DataFrame(source).to_csv(args.output / 'source.csv', index=False)
    (args.output / 'meta.json').write_text(json.dumps(
        {'protocol': __doc__, 'images': int(stack.shape[0]), 'patients': len(set(s['patient_id'] for s in source)),
         'label_columns': ['bmd_L1-L4', 'T_L1-L4', 'age', 'BMI', 'is_female'],
         'size': SIZE, 'skipped': skipped, 'licence': 'CC BY-NC 4.0, non-commercial only',
         'caveats': ['Lumbar radiographs, not DXA. Used only as a pretext task.',
                     'Label is DXA-measured density; no positioning or quality annotation exists.',
                     'CC BY-NC forbids commercial use of anything derived from it.']},
        ensure_ascii=False, indent=2) + '\n')
    print(f'\n{stack.shape} images, {target.shape[1]} labels, {len(set(s["patient_id"] for s in source))} patients')
    print(f'bmd  mean {target[:, 0].mean():.3f} sd {target[:, 0].std():.3f}')
    print(f'tscore mean {target[:, 1].mean():.3f} sd {target[:, 1].std():.3f}')
    print(f'skipped: {skipped}')
    (args.output / 'images.npy').write_bytes(stack.tobytes()) if False else None


if __name__ == '__main__':
    main()