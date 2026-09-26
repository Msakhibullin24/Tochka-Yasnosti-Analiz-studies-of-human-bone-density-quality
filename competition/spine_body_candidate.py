"""Alternate unnumbered spine body hypotheses in original pixel coordinates.

Compared with midpoint-between-valleys candidates, minima themselves can better
locate low-intensity trabecular interiors in bright-cortex exports. This remains
a hypothesis: discs, burned-in graphics and low contrast can also create minima.
"""
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

from dxaqc.geometry import spine_centerline


def detect(pixels, pixel_mm_y, pixel_mm_x):
    if pixels.ndim != 2 or not all(np.isfinite(v) and v > 0 for v in (pixel_mm_y, pixel_mm_x)):
        raise ValueError('Invalid spine raster or spacing')
    h, w = pixels.shape
    if h < 2 or w < 2:
        raise ValueError('Spine raster is empty or too small')
    physical_width = max(1, round(w*pixel_mm_x/pixel_mm_y))
    if h*physical_width > 36_000_000:
        raise ValueError('Physical grid exceeds image limit')
    ratio = physical_width/w
    physical = cv2.resize(pixels, (physical_width, h), interpolation=cv2.INTER_LINEAR) if physical_width != w else pixels
    ys, centre, _ = spine_centerline(physical)
    smooth = cv2.GaussianBlur(physical, (0, 0), 1.5)
    half = max(3, int(.07*physical_width))
    profile = np.array([smooth[y, max(0, int(x)-half):min(physical_width, int(x)+half+1)].mean()
                        for y, x in zip(ys, centre)])
    profile = cv2.GaussianBlur(profile.astype(np.float32).reshape(-1, 1), (1, 0),
                               sigmaY=2, sigmaX=0).ravel()
    positions, _ = find_peaks(-profile, distance=max(8, int(.08*h)),
                              prominence=max(4, float(np.std(profile))*.3))
    positions = positions[(positions > .03*h) & (positions < .92*h)]
    points = [[float((centre[y]+.5)/ratio-.5), float(y)] for y in positions]
    return {'method': 'central_intensity_minima_without_numbering',
            'coordinate_system': 'original_pixel_centres', 'points': points,
            'absolute_levels': None, 'clinical_validation': False,
            'limitations': ['Minima may correspond to disc spaces or graphics instead of vertebral interiors.',
                            'No anatomical level can be inferred by counting these points.']}


def stability(first, second, pixel_mm_y, pixel_mm_x):
    if not all(np.isfinite(v) and v > 0 for v in (pixel_mm_y, pixel_mm_x)):
        raise ValueError('Invalid stability spacing')
    a, b = np.asarray(first['points'], dtype=float), np.asarray(second['points'], dtype=float)
    if not len(a) or not len(b):
        return {'status': 'unavailable', 'max_shift_mm': None,
                'first_count': len(a), 'second_count': len(b)}
    if any(points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() for points in (a, b)):
        raise ValueError('Invalid candidate coordinates')
    scale = np.array([pixel_mm_x, pixel_mm_y])
    distances = np.linalg.norm((a[:, None, :]-b[None, :, :])*scale, axis=2)
    left, right = linear_sum_assignment(distances)
    largest = float(distances[left, right].max())
    return {'status': 'repeatable' if len(a)==len(b) and largest <= 5. else 'unstable',
            'max_shift_mm': largest, 'first_count': len(a), 'second_count': len(b),
            'technical_limit_mm': 5., 'clinical_validation': False}


if __name__ == '__main__':
    import argparse
    import csv
    import hashlib
    import json
    import warnings
    from pathlib import Path
    from dxaqc.dicom_io import read_dxa
    from source_integrity import inspect_sources
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'labels', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file')
    identity = inspect_sources([('organizer', args.labels, args.dataset)])
    with args.labels.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['region']=='spine' and r['quality_class'] in ('0', '1')]
    cases = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*')
        for row in rows:
            image = read_dxa(args.dataset/row['first_source_path'])
            sx = image.pixel_mm_x or image.pixel_mm
            base = detect(image.pixels, image.pixel_mm, sx)
            changed = np.clip(image.pixels.astype(float)*1.1-3, 0, 255).astype(np.uint8)
            perturbed = detect(changed, image.pixel_mm, sx)
            cases.append({'pixel_sha256': image.pixel_sha256, 'points': base['points'],
                          'stability': stability(base, perturbed, image.pixel_mm, sx)})
    report = {'scope': 'unnumbered candidate brightness repeatability; no clinical accuracy or anatomical numbering',
              'clinical_validation': False, 'images': len(cases),
              'source_identity': {k: v for k, v in identity.items() if k != 'entries'},
              'evaluation_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'counts': {status: sum(c['stability']['status']==status for c in cases)
                         for status in ('repeatable', 'unstable', 'unavailable')}, 'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'source_identity')}))
