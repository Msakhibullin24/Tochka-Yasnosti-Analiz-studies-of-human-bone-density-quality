"""Create an offline blind review package from source DICOMs and matching results.

No predictions, QC verdicts or source DICOM metadata are copied into the page.
Image pixels can contain burned-in labels; this is a local review tool, not an
anonymisation service. Source identity and decoding geometry are checked first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import cv2

from dxaqc.dicom_io import read_dxa
from dxaqc.validate_anatomy import read_rows, template


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(results, source, output):
    results, source, output = map(lambda p: Path(p).resolve(), (results, source, output))
    if not source.is_dir() or output.exists() or output.is_relative_to(source):
        raise ValueError('Use an existing source directory and a new output outside that directory')
    rows = read_rows(results)
    if not rows or len({r['image_uid'] for r in rows}) != len(rows):
        raise ValueError('Successful unique source images with unambiguous UIDs are required')
    reference = template(rows)
    reference.update(results_sha256=digest(results), review_package_version=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.anatomy-review-', dir=output.parent))
    try:
        (temporary/'images').mkdir()
        for number, (row, case) in enumerate(zip(rows, reference['cases'])):
            path = (source/row['path_to_file']).resolve()
            if not path.is_relative_to(source) or not path.is_file():
                raise ValueError('Source path escapes input or does not exist')
            image = read_dxa(path)
            height, width = image.pixels.shape
            if (image.image_uid != row['image_uid'] or width != int(row['image_width'])
                    or height != int(row['image_height'])):
                raise ValueError('Source identity or image geometry differs from the results')
            filename = f'images/{number:04d}.png'
            if not cv2.imwrite(str(temporary/filename), image.pixels):
                raise OSError('Failed to encode the review image')
            spine = 'Th12' in case['landmarks']
            case['named_regions'] = {name: {'visible': None, 'polygon': None}
                                     for name in (('Th12', 'L1', 'L2', 'L3', 'L4') if spine else
                                              ('femoral_neck_roi', 'femur', 'greater_trochanter',
                                               'lesser_trochanter', 'ischium'))}
            if spine:
                case['th12_full_body'] = {'status': None, 'quadrilateral': None}
            if not spine:
                case['rotation'] = None
                case['axes'] = {'femoral_neck_axis': {'visible': None, 'points': None}}
            case.update(image=filename, width=width, height=height,
                        source_sha256=digest(path), pixel_sha256=image.pixel_sha256,
                        study_group=hashlib.sha256(image.study_uid.encode()).hexdigest(),
                        pixel_mm_x=image.pixel_mm_x, pixel_mm_y=image.pixel_mm,
                        pixel_mm_source=image.pixel_mm_source)
        # Inline JSON supports file:// use without a web server or network.
        payload = json.dumps(reference, ensure_ascii=False).replace('<', '\\u003c')
        assets = Path(__file__).parent/'anatomy_review'
        page = (assets/'index.html').read_text().replace('__REVIEW_DATA__', payload)
        (temporary/'index.html').write_text(page)
        shutil.copy2(assets/'review.js', temporary/'review.js')
        (temporary/'template.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2)+'\n')
        (temporary/'manifest.json').write_text(json.dumps({
            'results_sha256': digest(results), 'cases': len(rows),
            'files_sha256': {str(p.relative_to(temporary)): digest(p)
                             for p in sorted(temporary.rglob('*')) if p.is_file()},
            'scope': 'blind source review; contains local image pixels, may contain burned-in labels; no model predictions'}, indent=2)+'\n')
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {'cases': len(rows), 'page': str(output/'index.html'), 'predictions_included': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.results, args.source, args.output), ensure_ascii=False))
        return 0
    except (ValueError, OSError) as exc:
        print(f'REVIEW_PACKAGE_FAILED: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
