"""Normalize Spinal-AI corner ordering without inventing anatomical level labels.

Publisher arrays use TL/TR/BL/BR, which self-intersect as COCO polygons. Sort
the same four points around their centroid; quarantine invalid coordinates.
Original annotations are never overwritten and semantic categories stay intact.
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np


def quad_polygon(points, width, height):
    values = np.asarray(points, dtype=float)
    if (values.shape != (8,) or not np.isfinite(values).all() or width <= 0 or height <= 0):
        raise ValueError('Expected four finite source corners')
    values = values.reshape(4, 2)
    if ((values < 0).any() or (values[:, 0] >= width).any() or (values[:, 1] >= height).any()
            or len(np.unique(values, axis=0)) != 4):
        raise ValueError('Invalid source corner coordinates')
    centered = values-values.mean(axis=0)
    ordered = values[np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))]
    edges = np.roll(ordered, -1, axis=0)-ordered
    next_edges = np.roll(edges, -1, axis=0)
    cross = edges[:, 0]*next_edges[:, 1]-edges[:, 1]*next_edges[:, 0]
    if not ((cross > 0).all() or (cross < 0).all()):
        raise ValueError('Source corners are not a nondegenerate convex quad')
    return ordered.ravel().tolist()


def normalize(data):
    images = {r['id']: r for r in data['images']}
    if len(images) != len(data['images']):
        raise ValueError('Duplicate image identifiers')
    accepted, rejected = [], []
    for annotation in data['annotations']:
        try:
            image = images[annotation['image_id']]
            polygons = annotation['segmentation']
            if len(polygons) != 1:
                raise ValueError('Expected one source quadrilateral')
            polygon = quad_polygon(polygons[0], image['width'], image['height'])
            xy = np.asarray(polygon).reshape(4, 2)
            area = float(abs((xy[:, 0]*np.roll(xy[:, 1], -1)
                              - xy[:, 1]*np.roll(xy[:, 0], -1)).sum())/2)
            accepted.append({**annotation, 'source_area': annotation.get('area'),
                             'area': area, 'segmentation': [polygon]})
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({'annotation_id': annotation.get('id'),
                             'image_id': annotation.get('image_id'), 'reason': str(exc)})
    return {**data, 'annotations': accepted,
            'conversion': {'corner_order': 'centroid-angle ordering; original points preserved',
                           'numbered_anatomical_labels': False,
                           'invalid_annotations_quarantined': len(rejected)}}, rejected


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output directory')
    with zipfile.ZipFile(args.archive) as archive:
        names = [n for n in archive.namelist() if n.endswith('.json')]
        if len(names) != 1:
            p.error('Expected one source COCO JSON')
        data = json.loads(archive.read(names[0]))
    converted, rejected = normalize(data)
    args.output.mkdir(parents=True)
    (args.output/'annotations.json').write_text(json.dumps(converted)+'\n')
    (args.output/'quarantine.json').write_text(json.dumps(rejected, indent=2)+'\n')
    print(json.dumps({'accepted': len(converted['annotations']), 'quarantined': len(rejected)}))
