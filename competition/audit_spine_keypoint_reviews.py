"""Check two independent spine COCO forms and report point differences in mm.

This never creates a consensus annotation or turns reviewed forms into training
labels. A human must adjudicate levels, visibility and point coordinates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from prepare_annotations import KEYPOINTS


def _reviewed(path: Path, ids: set[str]) -> dict[str, dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(ids) or {row['image_id'] for row in rows} != ids:
        raise ValueError('CSV review must cover every image exactly once')
    return {row['image_id']: row for row in rows}


def _annotations(path: Path, cases: list[dict]) -> dict[str, list[tuple[float, float, int]]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    categories = data.get('categories', [])
    if len(categories) != 1 or categories[0].get('keypoints') != KEYPOINTS['spine']:
        raise ValueError('COCO spine keypoint schema differs from the review packet')
    by_name = {case['image']: case for case in cases}
    images = data.get('images', [])
    if (len(images) != len(cases) or len({item['id'] for item in images}) != len(images)
            or {item['file_name'] for item in images} != set(by_name)):
        raise ValueError('COCO images do not cover the review packet exactly once')
    by_id = {}
    for item in images:
        case = by_name[item['file_name']]
        if (item['width'], item['height']) != (case['width'], case['height']):
            raise ValueError('COCO image geometry differs from the review packet')
        by_id[item['id']] = case
    result = {}
    for ann in data.get('annotations', []):
        if ann.get('image_id') not in by_id or ann.get('category_id') != categories[0]['id']:
            raise ValueError('COCO annotation refers to an unknown image or category')
        case = by_id[ann['image_id']]
        ident = case['image_id']
        if ident in result:
            raise ValueError('Only one whole-spine annotation per image is allowed')
        raw = ann.get('keypoints', [])
        if len(raw) != 3 * len(KEYPOINTS['spine']):
            raise ValueError('COCO annotation has an incomplete keypoint vector')
        points = []
        for index in range(0, len(raw), 3):
            x, y, visibility = raw[index:index + 3]
            if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y, visibility))
                    or visibility not in (0, 1, 2)):
                raise ValueError('COCO keypoint coordinates or visibility are invalid')
            if visibility and not (0 <= x < case['width'] and 0 <= y < case['height']):
                raise ValueError('Visible keypoint is outside the image')
            if not visibility and (x != 0 or y != 0):
                raise ValueError('Invisible keypoint must use (0, 0, 0)')
            points.append((float(x), float(y), int(visibility)))
        result[ident] = points
    return result


def audit(packet: Path, csv_a: Path, csv_b: Path, coco_a: Path, coco_b: Path) -> dict:
    manifest = json.loads((packet / 'manifest.json').read_text(encoding='utf-8'))
    cases = manifest['cases']
    if not cases or any(case['region'] != 'spine' for case in cases):
        raise ValueError('This audit requires a spine-only blind packet')
    ids = {case['image_id'] for case in cases}
    if len(ids) != len(cases):
        raise ValueError('Duplicate image IDs in review packet')
    reviews = (_reviewed(csv_a, ids), _reviewed(csv_b, ids))
    points = (_annotations(coco_a, cases), _annotations(coco_b, cases))
    results = []
    for case in cases:
        ident = case['image_id']
        left, right = (reviews[0][ident], reviews[1][ident])
        status = 'pending'
        differences = []
        if left.get('reviewed', '').lower() == right.get('reviewed', '').lower() == 'true':
            reviewer_a, reviewer_b = left.get('reviewer_id', '').strip(), right.get('reviewer_id', '').strip()
            if not reviewer_a or not reviewer_b or reviewer_a == reviewer_b:
                raise ValueError('Completed reviews require distinct named readers')
            a, b = points[0].get(ident), points[1].get(ident)
            if a is None and b is None and left.get('axis_assessability') == right.get('axis_assessability') == 'unassessable':
                status = 'unassessable_reviewed'
            elif a is None or b is None:
                status = 'missing_landmarks'
            else:
                status = 'requires_adjudication'
                mm_x, mm_y = float(case['pixel_mm_x']), float(case['pixel_mm_y'])
                if not math.isfinite(mm_x) or not math.isfinite(mm_y) or min(mm_x, mm_y) <= 0:
                    raise ValueError('Invalid physical pixel spacing in review packet')
                for name, pa, pb in zip(KEYPOINTS['spine'], a, b):
                    if pa[2] and pb[2]:
                        differences.append({'name': name, 'both_visible': True,
                                            'distance_mm': math.hypot((pa[0] - pb[0]) * mm_x,
                                                                      (pa[1] - pb[1]) * mm_y)})
                    elif bool(pa[2]) != bool(pb[2]):
                        differences.append({'name': name, 'both_visible': False,
                                            'visibility_disagreement': True})
        results.append({'image_id': ident, 'status': status,
                        'point_differences': differences})
    return {'scope': 'independent point review; no automatic consensus or training labels',
            'ground_truth_ready': False,
            'counts': dict(sorted(Counter(item['status'] for item in results).items())),
            'images': results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True, help='Extracted blind directory')
    parser.add_argument('--review-a', type=Path, required=True)
    parser.add_argument('--review-b', type=Path, required=True)
    parser.add_argument('--coco-a', type=Path, required=True)
    parser.add_argument('--coco-b', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new audit output file')
    result = audit(args.packet, args.review_a, args.review_b, args.coco_a, args.coco_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
