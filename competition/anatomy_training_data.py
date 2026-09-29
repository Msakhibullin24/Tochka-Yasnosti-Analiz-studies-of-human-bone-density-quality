"""Published annotation adapters with explicit target semantics and split identity."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from dxaqc.dicom_io import read_dxa
from evaluate_external_roi_masks import read_seg_nrrd
from evaluate_organizer_dataset import sha256
from spinal_ai_annotations import quad_polygon

SPINE_CLASSES = ['background'] + [f'Th{i}' for i in range(1, 13)] + [f'L{i}' for i in range(1, 6)]


def aasce_target(points, shape):
    h, w = shape
    points = np.asarray(points, dtype=float)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError('Expected 17 consecutive TL/TR/BL/BR quadrilaterals')
    target = np.zeros(shape, np.uint8)
    for level, corners in enumerate(points.reshape(17, 4, 2), 1):
        polygon = np.asarray(quad_polygon(corners.ravel(), w, h)).reshape(4, 2)
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        if np.any((target > 0) & (mask > 0)):
            raise ValueError('Overlapping vertebral annotations')
        target[mask > 0] = level
    centers = points.reshape(17, 4, 2).mean(axis=1)
    if np.any(np.diff(centers[:, 1]) <= 0):
        raise ValueError('Annotation order is not superior to inferior')
    return target


def dxa_spine_target(masks, shape):
    target = np.full(shape, -100, np.int16)
    foreground = np.zeros(shape, bool)
    for level in range(1, 5):
        mask = np.asarray(masks[f'Lumbar_{level}_bone_area'], dtype=bool)
        if mask.shape != shape or not mask.any() or np.any(foreground & mask):
            raise ValueError('Invalid named DXA ROI masks')
        foreground |= mask
    ys, xs = np.where(foreground)
    # Unknown anatomy outside annotated L1-L4 is ignored, not labeled absent.
    target[ys.min():ys.max()+1, max(0, xs.min()-4):min(shape[1], xs.max()+5)] = 0
    for level in range(1, 5):
        target[masks[f'Lumbar_{level}_bone_area']] = 12 + level
    return target


def records(aasce_root, dxa_root):
    rows, excluded, seen = [], [], set()
    images = {p.name: p for p in Path(aasce_root).rglob('*.jpg')}
    for annotation in sorted(Path(aasce_root).rglob('*.jpg.mat')):
        image = images.get(annotation.name[:-4])
        if image is None:
            raise ValueError('AASCE annotation has no paired image')
        pixels = cv2.imread(str(image), 0)
        try:
            if pixels is None:
                raise ValueError('Unreadable image')
            aasce_target(loadmat(annotation)['p2'], pixels.shape)
        except ValueError as exc:
            excluded.append({'annotation_sha256': sha256(annotation), 'reason': str(exc)})
            continue
        digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError('Exact source pixel duplicate')
        seen.add(digest)
        rows.append({'image': str(image), 'annotation': str(annotation), 'source': 'aasce',
                     'group': 'aasce:' + digest, 'group_scope': 'image; patient identity unavailable',
                     'image_sha256': sha256(image), 'annotation_sha256': sha256(annotation),
                     'pixel_sha256': digest, 'target_semantics': 'human vertebral quadrilateral'})
    for patient in sorted((Path(dxa_root) / 'Annotation').iterdir()):
        if not patient.is_dir():
            continue
        image = patient / 'images/spine_image.dcm'
        annotation = patient / 'segmentations/spine_image.seg.nrrd'
        raster = read_dxa(image)
        dxa_spine_target(read_seg_nrrd(annotation, raster.pixels.shape), raster.pixels.shape)
        digest = hashlib.sha256(str(raster.pixels.shape).encode() + raster.pixels.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError('Exact source pixel duplicate')
        seen.add(digest)
        rows.append({'image': str(image), 'annotation': str(annotation), 'source': 'ramathibodi',
                     'group': 'ramathibodi:' + hashlib.sha256(patient.name.encode()).hexdigest(),
                     'group_scope': 'publisher patient directory',
                     'image_sha256': sha256(image), 'annotation_sha256': sha256(annotation),
                     'pixel_sha256': digest, 'target_semantics': 'vendor named bone-area ROI; not bone contour'})
    if not rows:
        raise ValueError('No paired anatomical annotations')
    return rows, excluded


def load_record(row):
    image, annotation = Path(row['image']), Path(row['annotation'])
    if sha256(image) != row['image_sha256'] or sha256(annotation) != row['annotation_sha256']:
        raise ValueError('Source changed after inventory')
    if row['source'] == 'aasce':
        pixels = cv2.imread(str(image), 0)
        target = aasce_target(loadmat(annotation)['p2'], pixels.shape)
    else:
        pixels = read_dxa(image).pixels
        target = dxa_spine_target(read_seg_nrrd(annotation, pixels.shape), pixels.shape)
    if 'crop_xyxy' in row:
        x0,y0,x1,y1=row['crop_xyxy']
        if not (0<=x0<x1<=pixels.shape[1] and 0<=y0<y1<=pixels.shape[0]):
            raise ValueError('Derived crop escapes original raster')
        pixels,target=pixels[y0:y1,x0:x1].copy(),target[y0:y1,x0:x1].copy()
    return pixels, target


def lumbar_crop_bounds(target):
    """Derived view from labelled Th12--L5; not a predicted clinical crop.

    Keep the inferior half of annotated Th12 and all labelled lumbar bodies.
    The crop is defined on the training reference only, never on inference input.
    """
    target=np.asarray(target)
    if target.ndim!=2 or not all((target==level).any() for level in range(12,18)):
        raise ValueError('Named Th12--L5 annotations required for lumbar view')
    y12,x12=np.where(target==12)
    yl,xl=np.where((target>=13)&(target<=17))
    y0=int((y12.min()+y12.max()+1)//2)
    lumbar_height=int(yl.max()-y12.min()+1)
    padding=max(2,round(lumbar_height*.1))
    x0=max(0,int(min(x12.min(),xl.min()))-padding)
    x1=min(target.shape[1],int(max(x12.max(),xl.max()))+padding+1)
    y1=min(target.shape[0],int(yl.max())+padding+1)
    if x1-x0<8 or y1-y0<8:
        raise ValueError('Degenerate derived lumbar view')
    return [x0,y0,x1,y1]


def derive_lumbar_views(rows,test):
    """Parent split is inherited; derived views never count as new patients."""
    derived=[];flags=[]
    for row,held in zip(rows,test):
        derived.append({**row,'view':'original'});flags.append(bool(held))
        if row['source']=='aasce':
            _,target=load_record(row)
            derived.append({**row,'view':'lumbar_half_th12','crop_xyxy':lumbar_crop_bounds(target)})
            flags.append(bool(held))
    return derived,np.asarray(flags,dtype=bool)


def split_records(rows, seed=17):
    """Source-stratified groups; all derived views must inherit their parent split."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(rows), dtype=bool)
    for source in sorted({r['source'] for r in rows}):
        groups = sorted({r['group'] for r in rows if r['source'] == source})
        if len(groups) < 5:
            raise ValueError('Need at least five independent source groups')
        selected = set(rng.permutation(groups)[:max(1, len(groups)//5)])
        for i, row in enumerate(rows):
            test[i] = test[i] or row['group'] in selected
    if {r['group'] for r, t in zip(rows, test) if t} & {r['group'] for r, t in zip(rows, test) if not t}:
        raise ValueError('Training/test group leakage')
    return test
