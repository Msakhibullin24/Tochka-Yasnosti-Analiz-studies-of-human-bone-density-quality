"""Read explicit DICOM ROI without interpreting arbitrary graphics as anatomy.

Coordinates are original image pixel centres (zero based). Overlay origins are
one based; GSPS PIXEL coordinates refer to pixel edges, hence the -0.5 offset.
See DICOM PS3.3 C.9.2 and C.10.5. Never silently replace absent ROI by a crop.
"""
from __future__ import annotations

import cv2
import numpy as np

MAX_POINTS = 32768


def extract_roi(ds, shape, image_uid):
    h, w = shape
    result = {'status': 'absent', 'rois': [], 'issues': []}
    for group in range(0x6000, 0x6020, 2):
        if (group, 0x3000) not in ds:
            continue
        if str(ds.get((group, 0x0040), '').value if (group, 0x0040) in ds else '') != 'R':
            result['issues'].append(f'{group:04x}: graphics overlay is not an explicit ROI')
            continue
        try:
            rows, cols = int(ds[group, 0x0010].value), int(ds[group, 0x0011].value)
            if rows <= 0 or cols <= 0 or rows * cols > 36_000_000:
                raise ValueError('overlay dimensions exceed limit')
            if int(ds.get((group, 0x0015)).value if (group, 0x0015) in ds else 1) != 1:
                raise ValueError('multi-frame overlay is unsupported')
            if int(ds[group, 0x0100].value) != 1 or int(ds[group, 0x0102].value) != 0:
                raise ValueError('embedded overlay bits are unsupported')
            origin = [int(x) - 1 for x in ds[group, 0x0050].value]
            mask = ds.overlay_array(group).astype(np.uint8)
            if mask.shape != (rows, cols):
                raise ValueError('overlay dimensions disagree')
            ys, xs = np.where(mask)
            if not len(xs):
                raise ValueError('empty ROI')
            if min(xs) + origin[1] < 0 or max(xs) + origin[1] >= w or min(ys) + origin[0] < 0 or max(ys) + origin[0] >= h:
                raise ValueError('ROI contains pixels outside source image')
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if sum(len(c) for c in contours) > MAX_POINTS:
                raise ValueError('ROI contour exceeds point limit')
            outlines = []
            for i, c in enumerate(contours):
                points = c[:, 0].astype(float) + [origin[1], origin[0]]
                parent, depth = hierarchy[0, i, 3], 0
                while parent >= 0:
                    depth += 1
                    parent = hierarchy[0, parent, 3]
                outlines.append({'points': points.tolist(), 'depth': depth})
            # Exact raster membership, independent of contour display semantics.
            flat = mask.reshape(-1)
            edges = np.diff(np.r_[0, flat, 0].astype(np.int8))
            starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
            if len(starts) > 65536:
                raise ValueError('ROI raster run count exceeds limit')
            raster = {'shape': [rows, cols], 'origin': origin,
                      'runs': np.column_stack((starts, stops)).tolist()}
            result['rois'].append({'raster': raster, 'id': f'overlay-{group:04x}', 'source': 'DICOM_OVERLAY_R',
                                  'source_uid': str(ds.get('SOPInstanceUID', '')),
                                  'coordinate_system': 'original_pixel_centres',
                                  'contours': outlines, 'area_pixels': int(len(xs)),
                                  'bounds': [int(min(xs)+origin[1]), int(min(ys)+origin[0]), int(max(xs)+origin[1]), int(max(ys)+origin[0])]})
        except (ValueError, KeyError, TypeError, AttributeError, IndexError, RuntimeError) as exc:
            result['issues'].append(f'{group:04x}: {exc}')
    # Only explicitly named ROI layers are interpreted as measurement regions.
    # Other closed graphics can be frames, text boxes or decoration.
    for i, annotation in enumerate(ds.get('GraphicAnnotationSequence', [])):
        layer = str(annotation.get('GraphicLayer', '')).upper()
        if not (layer == 'ROI' or layer.startswith('ROI_')):
            result['issues'].append(f'graphic-{i}: layer is not explicitly named ROI')
            continue
        refs = annotation.get('ReferencedImageSequence', [])
        if not refs:
            refs = [ref for series in ds.get('ReferencedSeriesSequence', []) for ref in series.get('ReferencedImageSequence', [])]
        if refs and not any(str(ref.get('ReferencedSOPInstanceUID', '')) == image_uid for ref in refs):
            continue
        if not refs and str(ds.get('SOPInstanceUID', '')) != image_uid:
            result['issues'].append(f'graphic-{i}: missing source image reference')
            continue
        if any('ReferencedFrameNumber' in ref for ref in refs):
            result['issues'].append(f'graphic-{i}: frame-specific annotation unsupported')
            continue
        for j, graphic in enumerate(annotation.get('GraphicObjectSequence', [])):
            try:
                if str(graphic.get('GraphicAnnotationUnits', '')) != 'PIXEL':
                    raise ValueError('only image-relative PIXEL units supported')
                if str(graphic.get('GraphicType', '')) != 'POLYLINE' or int(graphic.get('GraphicDimensions', 0)) != 2:
                    raise ValueError('only closed two-dimensional POLYLINE ROI supported')
                values = np.asarray(graphic.get('GraphicData', []), dtype=float)
                if values.size < 8 or values.size > MAX_POINTS * 2 or values.size % 2 or not np.isfinite(values).all():
                    raise ValueError('invalid ROI coordinate array')
                points = values.reshape(-1, 2)
                if len(points) != int(graphic.get('NumberOfGraphicPoints', 0)) or not np.array_equal(points[0], points[-1]):
                    raise ValueError('ROI must be a closed polygon with matching point count')
                if np.any(points < 0) or np.any(points > [w, h]):
                    raise ValueError('ROI outside source image')
                points = points[:-1] - 0.5
                if cv2.contourArea(points.astype(np.float32)) <= 0:
                    raise ValueError('zero area ROI')
                result['rois'].append({'id': f'graphic-{i}-{j}', 'source': 'DICOM_GRAPHIC_ROI',
                                      'source_uid': str(ds.get('SOPInstanceUID', '')),
                                      'coordinate_system': 'original_pixel_centres',
                                      'contours': [{'points': points.tolist(), 'depth': 0}],
                                      'bounds': [*points.min(0).tolist(), *points.max(0).tolist()]})
            except (ValueError, TypeError, AttributeError) as exc:
                result['issues'].append(f'graphic-{i}-{j}: {exc}')
    result['status'] = ('partial' if result['issues'] else 'extracted') if result['rois'] else ('unavailable' if result['issues'] else 'absent')
    return result


def contains(roi, point):
    """Nested contours retain holes and disconnected regions."""
    if 'raster' in roi:
        import bisect
        raster = roi['raster']
        # Pixel centres use nearest-neighbour membership; outer half-pixel
        # boundaries are half-open. No contour simplification can remove a pixel.
        x = int(np.floor(float(point[0]) + .5)) - raster['origin'][1]
        y = int(np.floor(float(point[1]) + .5)) - raster['origin'][0]
        h, w = raster['shape']
        if not (0 <= x < w and 0 <= y < h):
            return False
        index = y * w + x
        runs = raster['runs']
        pos = bisect.bisect_right(runs, [index, float('inf')]) - 1
        return pos >= 0 and runs[pos][0] <= index < runs[pos][1]
    inside = [c['depth'] for c in roi['contours'] if len(c['points']) >= 3 and
              cv2.pointPolygonTest(np.asarray(c['points'], np.float32), tuple(map(float, point)), False) >= 0]
    return bool(inside) and max(inside) % 2 == 0


def presentation_states(files):
    """Index referenced GSPS objects; unreferenced/unsupported files stay in input.

    Parsing errors are left to the normal per-file failure path. Do not discard
    an annotation object whose source image is missing from this batch.
    """
    import warnings
    import pydicom
    headers, images, by_uid = {}, set(), {}
    for path in files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            uid = str(ds.get('SOPInstanceUID', ''))
            if str(ds.get('SOPClassUID', '')) == '1.2.840.10008.5.1.4.1.1.11.1':
                headers[path] = ds
            elif uid:
                images.add(uid)
        except Exception:
            continue
    consumed = set()
    for path, ds in headers.items():
        refs = {str(e.value) for e in ds.iterall() if e.keyword == 'ReferencedSOPInstanceUID'}
        if not refs or not refs <= images or 'GraphicAnnotationSequence' not in ds:
            continue
        for uid in sorted(refs):
            by_uid.setdefault(uid, []).append(ds)
        consumed.add(path)
    return [p for p in files if p not in consumed], by_uid


def merge_presentation_roi(img, states):
    """Annotations belong to the source instance, never to its pixel-cache twin."""
    result = {'rois': list(img.source_roi['rois']), 'issues': list(img.source_roi['issues'])}
    for ds in states:
        prefix = str(ds.get('SOPInstanceUID', ''))
        if str(ds.get('StudyInstanceUID', '')) != img.study_uid:
            result['issues'].append(f'{prefix}: annotation belongs to a different study')
            continue
        value = extract_roi(ds, img.pixels.shape, img.image_uid)
        result['rois'].extend([{**r, 'id': prefix + '/' + r['id']} for r in value['rois']])
        result['issues'].extend([prefix + ': ' + x for x in value['issues']])
    result['status'] = ('partial' if result['issues'] else 'extracted') if result['rois'] else ('unavailable' if result['issues'] else 'absent')
    return result
