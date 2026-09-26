"""Orthographic CT/mask projection around the segmented femur's PCA shaft axis.

Requires optional nibabel in the research environment. Relative pose is known;
clinical rotation quality and DXA material decomposition remain unknown.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import affine_transform, zoom

from evaluate_organizer_dataset import sha256


def rotation_matrix(axis, degrees):
    axis = np.asarray(axis, float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8 or not np.isfinite(degrees):
        raise ValueError('Invalid shaft axis or rotation')
    axis = axis/np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    radians = np.radians(degrees)
    return np.eye(3)*np.cos(radians)+(1-np.cos(radians))*np.outer(axis, axis)+np.sin(radians)*cross


def shaft_axis(mask):
    indices = np.argwhere(mask)
    if len(indices) < 100:
        raise ValueError('Insufficient segmented bone for shaft estimation')
    sample = indices[::max(1, len(indices)//10000)].astype(float)
    center = sample.mean(0)
    values, vectors = np.linalg.eigh(np.cov((sample-center).T))
    axis = vectors[:, values.argmax()]
    if axis[2] < 0:
        axis = -axis
    return center, axis


def project(volume, mask, degrees, view):
    if volume.shape != mask.shape or volume.ndim != 3 or view not in ('frontal', 'lateral'):
        raise ValueError('Invalid paired 3D volume or view')
    center, axis = shaft_axis(mask)
    inverse = rotation_matrix(axis, degrees).T
    offset = center-inverse @ center
    rotated = affine_transform(volume, inverse, offset=offset, order=1, cval=-1000., prefilter=False)
    labels = affine_transform(mask.astype(np.uint8), inverse, offset=offset, order=0, cval=0, prefilter=False)
    beam = 1 if view == 'frontal' else 0
    # Approximate line integral, not a two-energy DXA simulation.
    attenuation = np.clip((rotated+1000.)/1000., 0, 4).sum(axis=beam).T[::-1].copy()
    image = cv2.normalize(attenuation, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    projected = labels.any(axis=beam).T[::-1].copy()
    return image, projected, {'shaft_axis_ras': axis.tolist(), 'center_voxels': center.tolist()}


def run(root, output):
    import nibabel as nib
    if output.exists():
        raise ValueError('Choose a new output directory')
    output.mkdir(parents=True)
    cases = []
    for patient in sorted(Path(root).glob('subject*')):
        image_path, mask_path = patient/'volume.nii.gz', patient/'mask.nii.gz'
        image = nib.as_closest_canonical(nib.load(image_path))
        mask_image = nib.as_closest_canonical(nib.load(mask_path))
        if image.shape != mask_image.shape or not np.allclose(image.affine, mask_image.affine, atol=1e-4):
            raise ValueError('CT and segmentation world grids differ')
        scale = np.array(image.header.get_zooms()[:3])/3.
        volume = zoom(image.get_fdata(dtype=np.float32), scale, order=1, prefilter=False)
        mask = zoom(np.asarray(mask_image.dataobj), scale, order=0, prefilter=False) > 0
        for degrees in (-30, -15, 0, 15, 30):
            for view in ('frontal', 'lateral'):
                pixels, projected, pose = project(volume, mask, degrees, view)
                stem = f'{patient.name}_{view}_{degrees:+d}'
                cv2.imwrite(str(output/f'{stem}.png'), pixels)
                cv2.imwrite(str(output/f'{stem}_mask.png'), projected.astype(np.uint8)*255)
                cases.append({'image': f'{stem}.png', 'mask': f'{stem}_mask.png',
                              'patient_group': patient.name, 'split': 'test' if patient.name == 'subject05' else 'train',
                              'view': view, 'relative_shaft_rotation_deg': degrees, 'pixel_mm': 3., **pose,
                              'ct_sha256': sha256(image_path), 'mask_sha256': sha256(mask_path),
                              'target_origin': 'synthetic_3d_pose', 'qc_targets': {},
                              'clinical_rotation_label': None})
        print(f'Rendered {patient.name}', flush=True)
    report = {'cases': cases, 'projection': 'orthographic attenuation approximation in canonical RAS',
              'code_sha256': sha256(Path(__file__)), 'clinical_validation': False,
              'limitations': ['Known relative pose does not prove clinically correct baseline rotation.',
                              'PCA shaft estimation can be biased by whole-bone shape.',
                              'Not a cone-beam or two-energy DXA simulation.',
                              'Femur source licence is CC BY-NC-ND 4.0; do not redistribute derived images.']}
    (output/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.root, args.output)
