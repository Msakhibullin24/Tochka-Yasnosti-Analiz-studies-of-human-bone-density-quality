"""Generate synthetic femur/pelvis views with independently varied camera and leg.

Camera AP is defined by the annotated anterior pelvic plane, not native CT Z.
No generated view is a clinical DXA projection/rotation label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import sys

import cv2
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.prepare_vsd_hip_projection import COMMIT, LICENSE, sphere_center
from experiments.train_silhouette_heatmaps import normalize_proximal_mask


def unit(vector):
    vector = np.asarray(vector, float)
    length = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or length < 1e-8:
        raise ValueError('Degenerate anatomical coordinate frame')
    return vector / length


def pelvic_frame(points):
    """Rows are leftward, anterior-normal, superior; determinant +1."""
    points = np.asarray(points, float)
    left = unit(points[2] - points[3])
    superior = (points[2] + points[3] - points[6] - points[7]) / 2
    superior = unit(superior - left * (superior @ left))
    anterior = unit(np.cross(superior, left))
    return np.stack([left, anterior, superior])


def rotate_about(points, center, axis, degrees):
    axis = unit(axis)
    delta = np.asarray(points) - center
    theta = np.radians(degrees)
    return (delta * np.cos(theta) + np.cross(axis, delta) * np.sin(theta)
            + (delta @ axis)[..., None] * axis * (1 - np.cos(theta)) + center)


def project(points, center, frame, camera_degrees):
    local = (np.asarray(points) - center) @ frame.T
    theta = np.radians(camera_degrees)
    horizontal = local[..., 0] * np.cos(theta) + local[..., 1] * np.sin(theta)
    return np.stack([horizontal * 2 + 199.5, -local[..., 2] * 2 + 199.5], -1)


def raster(vertices, faces, center, frame, camera_degrees):
    coordinates = project(vertices, center, frame, camera_degrees)
    triangles = coordinates[faces]
    visible = np.all(triangles.max(1) >= 0, axis=1) & np.all(triangles.min(1) < 400, axis=1)
    mask = np.zeros((400, 400), np.uint8)
    for triangle in np.rint(triangles[visible]).astype(np.int32):
        cv2.fillConvexPoly(mask, triangle, 1)
    return mask


def annotation(root, kind, subject, side=None):
    values = []
    names = None
    for r in range(1, 6):
        rows = loadmat(root / f'ManualLandmarks/{kind}/Rater{r}.mat', simplify_cells=True)['Results']
        matched = [row for row in rows if str(row[4]) == subject and (side is None or str(row[5]) == side)]
        if len(matched) != 1:
            raise ValueError('Ambiguous published annotation identity')
        for cycle in matched[0][:4]:
            if names is not None and list(cycle[:, 0]) != names:
                raise ValueError('Published annotation ordering differs')
            names = list(cycle[:, 0])
            values.append(np.vstack(cycle[:, 1]).astype(float))
    return np.median(values, axis=0)


def normalized_channels(femur, pelvis, sx=.5, sy=.5):
    normalized, affine = normalize_proximal_mask(femur, sx, sy)
    context = cv2.warpAffine(pelvis, affine, (320, 320), flags=cv2.INTER_NEAREST)
    field = cv2.warpAffine(np.ones_like(femur), affine, (320, 320), flags=cv2.INTER_NEAREST)
    return np.stack([cv2.resize(m, (128, 128), interpolation=cv2.INTER_NEAREST) for m in (normalized, context, field)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new dataset directory')
    inventory = json.loads((args.source / 'acquisition.json').read_text())
    pelvis_inventory = json.loads((args.source / 'pelvis-acquisition.json').read_text())
    if inventory['commit'] != COMMIT or inventory['data_license'] != LICENSE:
        raise ValueError('Unexpected pinned data source')
    for record in inventory['files'] + pelvis_inventory['files']:
        if hashlib.sha256((args.source / record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Published source checksum differs')
    rows = loadmat(args.source / 'ManualLandmarks/Femur/Rater1.mat', simplify_cells=True)['Results']
    subjects = sorted((str(row[4]), str(row[5])) for row in rows)
    args.output.mkdir(parents=True)
    records, inputs = [], []
    started = perf_counter()
    for subject, side in subjects:
        bones = {b['name']: b['mesh'] for b in loadmat(args.source / f'Bones/{subject}.mat', simplify_cells=True)['B']}
        femur, pelvis = bones[f'Femur_{side}'], bones[f'Hip_{side}']
        vertices = np.asarray(femur['vertices'], float)
        femur_faces = np.asarray(femur['faces'], int) - 1
        pelvis_vertices = np.asarray(pelvis['vertices'], float)
        pelvis_faces = np.asarray(pelvis['faces'], int) - 1
        femoral_points = annotation(args.source, 'Femur', subject, side)
        frame = pelvic_frame(annotation(args.source, 'Pelvis', subject))
        head = sphere_center(femoral_points[:6])
        distal = femoral_points[[13, 14, 20, 21]].mean(0)
        center = head - frame[2] * 45
        # Restrict triangles in physical superior coordinates before rasterization.
        relative_height = (vertices - head) @ frame[2]
        femur_faces = femur_faces[np.all(relative_height[femur_faces] > -170, axis=1)]
        pelvis_by_camera = {a: raster(pelvis_vertices, pelvis_faces, center, frame, a)
                            for a in (-90, -60, -30, 0, 30, 60, 90)}
        for rotation in (-30, -15, 0, 15, 30):
            rotated = rotate_about(vertices, head, head - distal, rotation)
            for camera in (-90, -60, -30, 0, 30, 60, 90):
                femur_mask = raster(rotated, femur_faces, center, frame, camera)
                channels = normalized_channels(femur_mask, pelvis_by_camera[camera])
                _, affine = normalize_proximal_mask(femur_mask, .5, .5)
                landmarks3d = np.vstack([femoral_points[10], femoral_points[11], head, femoral_points[6:10].mean(0)])
                landmarks3d = rotate_about(landmarks3d, head, head - distal, rotation)
                landmarks = project(landmarks3d, center, frame, camera) @ affine[:, :2].T + affine[:, 2]
                landmarks = (landmarks + .5) * 128 / 320 - .5
                if side == 'R':
                    channels = channels[..., ::-1].copy()
                    landmarks[:, 0] = 127 - landmarks[:, 0]
                inputs.append(channels)
                records.append({'subject': subject, 'side': side, 'camera_degrees': camera,
                                'leg_rotation_about_mechanical_axis_degrees': rotation,
                                'split': 'test' if subject in {s for s, _ in subjects[-4:]} else 'train',
                                'view': 'frontal' if camera == 0 else 'lateral' if abs(camera) == 90 else 'intermediate',
                                'landmark_points_128': landmarks.tolist(),
                                'normalized_pixel_mm': float(.5 / affine[0, 0] * 320 / 128),
                                'frame_determinant': float(np.linalg.det(frame))})
        print(json.dumps({'subject': subject, 'cases': len(records), 'seconds': perf_counter() - started}), flush=True)
    array_path = args.output / 'masks.npy'
    np.save(array_path, np.stack(inputs).astype(np.uint8))
    report = {'source_commit': COMMIT, 'data_license': LICENSE, 'records': records,
              'input_shape': list(np.stack(inputs).shape),
              'masks_sha256': hashlib.sha256(array_path.read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'clinical_validation': False,
              'protocol': 'APP camera frame from bilateral ASIS/pubis; 5 independent mechanical-axis leg rotations; 15/4 donor split',
              'limitations': ['Silhouettes, not DXA attenuation or measured segmentation noise.',
                              'Synthetic AP/lateral endpoints; no clinical threshold for intermediate views.',
                              'Leg zero is original donor pose, not certified neutral rotation.',
                              'Same previously inspected four held-out donors.']}
    (args.output / 'dataset.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'cases': len(records), 'seconds': perf_counter() - started}), flush=True)


if __name__ == '__main__':
    main()
