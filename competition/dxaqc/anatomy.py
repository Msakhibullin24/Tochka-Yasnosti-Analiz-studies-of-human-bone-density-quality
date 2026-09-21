"""Inspectable anatomical candidates and coverage checks, separate from quality scores.

These deterministic detectors are not expert-validated landmark models. Missing
or ambiguous evidence is never converted to a verified anatomical pass.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import find_peaks

from .geometry import spine_centerline
from .source_roi import contains


def _candidate(name, points, method):
    return {'name': name, 'status': 'candidate' if points else 'not_localized',
            'points': [[float(x), float(y)] for x, y in points], 'method': method, 'verified': False}


def detect_landmarks(pixels, region, overlay, pixel_mm_y=1., pixel_mm_x=1.):
    """All returned coordinates are in the original image, including left hips."""
    h, w = pixels.shape
    if not all(np.isfinite(v) and v > 0 for v in (pixel_mm_x, pixel_mm_y)):
        raise ValueError("invalid landmark calibration")
    width = max(1, round(w * pixel_mm_x / pixel_mm_y))
    if width * h > 36_000_000:
        raise ValueError("landmark physical grid exceeds image limit")
    if width != w:
        ratio = width / w
        physical = cv2.resize(pixels, (width, h), interpolation=cv2.INTER_LINEAR)
        mapped = {k: [((x+.5)*ratio-.5, y) for x,y in points] for k,points in overlay.items()}
        result = detect_landmarks(physical, region, mapped)
        for item in result['landmarks']:
            item['points'] = [[(x+.5)/ratio-.5, y] for x,y in item['points']]
            item['method'] = 'physical_grid:' + item['method']
        return result
    out = []
    if region == 'spine':
        ys, centre, _ = spine_centerline(pixels)
        smooth = cv2.GaussianBlur(pixels, (0, 0), 1.5)
        threshold = max(40, cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
        # Only connected lateral bone in the lower field is a crest candidate.
        for side in ('left', 'right'):
            mask = (smooth > threshold).astype(np.uint8)
            xx = np.arange(w)[None, :]
            mask[(xx >= centre[:, None] - .20*w) if side == 'left' else (xx <= centre[:, None] + .20*w)] = 0
            mask[:int(.55*h)] = 0
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            possible = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= max(20, .001*h*w)
                        and stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= .90*h]
            points = []
            if possible:
                component = max(possible, key=lambda i: stats[i, cv2.CC_STAT_AREA])
                y = int(stats[component, cv2.CC_STAT_TOP])
                points = [(float(np.where(labels[y] == component)[0].mean()), y)]
            out.append(_candidate('iliac_crest_' + side, points, 'lower_connected_lateral_bone'))
        # Disc-like intensity valleys delineate candidate bodies. Absolute vertebral
        # numbering cannot be inferred merely by counting from a cropped frame edge.
        half = max(3, int(.07*w))
        profile = np.array([smooth[y, max(0,int(x)-half):min(w,int(x)+half+1)].mean() for y,x in zip(ys,centre)])
        profile = cv2.GaussianBlur(profile.astype(np.float32).reshape(-1,1), (1,0), sigmaY=2, sigmaX=0).ravel()
        valleys, _ = find_peaks(-profile, distance=max(8, int(.08*h)), prominence=max(4, float(np.std(profile))*.3))
        valleys = valleys[(valleys > .03*h) & (valleys < .92*h)]
        bodies = []
        for first, last in zip(valleys[:-1], valleys[1:]):
            if .06*h <= last-first <= .25*h:
                y = (first+last)/2
                bodies.append((float(centre[int(y)]), float(y)))
        out.append(_candidate('vertebral_body_candidates', bodies, 'central_intensity_valleys_without_numbering'))
        out.append(_candidate('Th12', [], 'requires_independent_vertebral_numbering'))
    else:
        canonical = pixels[:, ::-1].copy() if region == 'hip_left' else pixels
        def original(points):
            return [(w-1-x if region == 'hip_left' else x, y) for x,y in points]
        for name, key in [('greater_trochanter','greater_trochanter_top'),('lesser_trochanter','lesser_trochanter'),('ischium','ischium_bottom')]:
            pts = [(x,y) for x,y in overlay.get(key, []) if 0 <= x < w and 0 <= y < h]
            out.append(_candidate(name, original(pts), 'tracked_bone_contour'))
        # Neck candidate: minimum thickness on the proximal femur component,
        # following diagonal sections perpendicular to the shaft-to-head direction.
        smooth = cv2.GaussianBlur(canonical, (0,0), 1.5)
        threshold = max(30, .8*cv2.threshold(smooth,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0])
        mask = (smooth > threshold).astype(np.uint8)
        axis = overlay.get('shaft_axis', [])
        neck = []
        if len(axis) == 2:
            sx = float(axis[-1][0])
            # Recover the proximal lateral cap from the femur component connected
            # to the distal shaft; a contour gap must not move its top down to
            # the most lateral shaft point. This does not alter model features.
            count, components = cv2.connectedComponents(mask)
            seed = components[int(.85*h):, max(0,int(sx-.08*w)):min(w,int(sx+.08*w)+1)]
            ids, sizes = np.unique(seed[seed>0], return_counts=True)
            femur = np.zeros_like(mask, dtype=bool)
            if len(ids):
                femur = components == ids[np.argmax(sizes)]
                yy,xx = np.where(femur)
                dx = axis[-1][0]-axis[0][0];dy = axis[-1][1]-axis[0][1]
                shaft_x = axis[0][0] + dx*(yy-axis[0][1])/max(abs(dy),1)
                lateral = (xx < shaft_x) & (yy < .7*h)
                if lateral.any():
                    top = int(yy[lateral].min())
                    cap = xx[lateral & (yy <= top+1)]
                    greater = next(item for item in out if item['name']=='greater_trochanter')
                    greater['points'] = [[float(x),float(y)] for x,y in original([(float(np.median(cap)),top)])]
                    greater['method'] = 'distal_shaft_connected_lateral_cap'
                    greater['status'] = 'candidate'
                    # The femoral head can merge with pelvis pixels that reach the
                    # top edge. Only a top-edge path in the lateral cap corridor
                    # makes this particular candidate ambiguous.
                    lateral_edge = min(w, int(np.median(cap) + .1*w) + 1)
                    if np.any(femur[0, :lateral_edge]):
                        greater.update(points=[], status='not_localized',
                                       method='ambiguous_component_at_image_boundary')
            candidates = []
            # Require two background boundaries; image edges or pelvis merges
            # must not be interpreted as a measured neck width.
            for y in range(int(.15*h),int(.55*h)):
                for x in range(max(1,int(sx+.04*w)),min(w-1,int(sx+.38*w)),max(1,w//80)):
                    if not femur[y,x]: continue
                    reach = int(min(h,w)*.25)
                    t = np.arange(-reach,reach+1)
                    xx, yy = np.rint(x+t*.707).astype(int), np.rint(y+t*.707).astype(int)
                    valid = (xx>=0)&(xx<w)&(yy>=0)&(yy<h)
                    line=np.zeros(len(t),np.uint8);line[valid]=femur[yy[valid],xx[valid]]
                    left=np.where(line[:reach]==0)[0];right=np.where(line[reach+1:]==0)[0]
                    if not len(left) or not len(right):continue
                    a,b=int(left[-1]),int(reach+1+right[0])
                    if not valid[a] or not valid[b]:continue
                    width=b-a
                    if .06*w <= width <= .22*w and abs((a+b)/2-reach) < 2:
                        candidates.append((width,x,y))
            if candidates:
                _,x,y=min(candidates)
                neck=original([(x,y)])
        out.append(_candidate('femoral_neck', neck, 'proximal_diagonal_thickness_candidate'))
    # Also covers the fallback contour when no shaft-connected component exists.
    for landmark in out:
        if landmark['name'] == 'greater_trochanter' and any(y <= 1 for _, y in landmark['points']):
            landmark.update(points=[], status='not_localized', method='ambiguous_component_at_image_boundary')
    return {'status': 'needs_validation', 'coordinate_system': 'original_pixel_centres',
            'landmarks': out, 'clinical_validation': False}


def assess_projection(pixels, region, landmarks, declared=''):
    """Frontal appearance hypothesis, not a validated AP/lateral classifier."""
    declared = declared.upper()
    evidence = {}
    if region == 'spine':
        h,w = pixels.shape
        _,centre,_=spine_centerline(pixels)
        offsets=np.arange(max(2,int(.08*w)),max(3,int(.26*w)))
        yy=np.arange(int(.15*h),int(.80*h))
        left=pixels[yy[:,None],np.clip(centre[yy,None].astype(int)-offsets,0,w-1)].astype(float)
        right=pixels[yy[:,None],np.clip(centre[yy,None].astype(int)+offsets,0,w-1)].astype(float)
        denom=float(np.mean(left+right))
        asymmetry=float(np.mean(abs(left-right))/max(denom,1))
        evidence={'bilateral_asymmetry':round(asymmetry,4),'lateral_signal':round(denom/2,3)}
        frontal=asymmetry<.35 and denom>25
    else:
        available={x['name'] for x in landmarks['landmarks'] if x['points']}
        frontal={'greater_trochanter','femoral_neck','ischium'} <= available
        evidence={'located_candidates':sorted(available)}
    status='candidate' if frontal else 'undetermined'
    return {'value':'frontal' if frontal else 'unknown', 'status':status,
            'method':'pixel_morphology_v1', 'evidence':evidence, 'declared_view_position':declared,
            'clinical_validation':False,
            'reason':'Предположение по структурам изображения; распознавание боковой проекции не валидировано.'}


def evaluate_source_roi(source, landmarks, img, region):
    checks=[]
    for roi in source['rois']:
        x0,y0,x1,y1=roi['bounds'];h,w=img.pixels.shape
        sx=img.pixel_mm_x or img.pixel_mm
        edge='right' if region=='hip_left' else 'left'
        margins={'top':y0*img.pixel_mm,'bottom':(h-1-y1)*img.pixel_mm,
                 'side':((w-1-x1) if edge=='right' else x0)*sx}
        known_scale=img.pixel_mm_source not in ('device_default','',None)
        required=['greater_trochanter','femoral_neck','ischium'] if region!='spine' else ['Th12','iliac_crest_left','iliac_crest_right']
        covered={}
        for item in landmarks['landmarks']:
            if item['name'] in required:
                covered[item['name']]=None if not item['points'] else all(contains(roi,p) for p in item['points'])
        # Different clinical ROIs (neck-only, total hip, individual vertebrae) have
        # different inclusion rules. Report relationships without guessing intent.
        checks.append({'roi_id':roi['id'], 'status':'needs_review', 'anatomical_validity':'undetermined',
                       'candidate_landmarks_inside':covered,
                       'margins_mm':{k:round(v,3) for k,v in margins.items()} if region!='spine' else None,
                       'margin_status':('pass' if all(margins[k]>=v-1e-8 for k,v in {'top':30,'bottom':30,'side':20}.items()) else 'fail') if known_scale and region!='spine' else 'unavailable',
                       'scale_source':img.pixel_mm_source,'edge':edge,
                       'reason':'Назначение исходной ROI и анатомические ориентиры требуют подтверждения; попадание кандидатов не доказывает корректность разметки.'})
    return {'status':source['status'],'rois':source['rois'],'issues':source['issues'],'checks':checks}


def image_assessment(img, result):
    """Shared single-image and batch contract. Cache only pixel-derived evidence."""
    if 'anatomy_candidates' not in result:
        result['anatomy_candidates'] = detect_landmarks(img.pixels, result['region'], result['overlay'], img.pixel_mm, img.pixel_mm_x or img.pixel_mm)
        result['projection_pixels'] = assess_projection(img.pixels, result['region'], result['anatomy_candidates'])
    projection = {**result['projection_pixels'], 'declared_view_position': img.view_position}
    return {'projection': projection, 'anatomy': result['anatomy_candidates'],
            'source_roi': evaluate_source_roi(img.source_roi, result['anatomy_candidates'], img, result['region']),
            'complete': False, 'version': '2'}


def display_overlay(result, width):
    """One current set of hip candidates for UI and SC; model features stay frozen."""
    overlay = dict(result['overlay'])
    if result['region'] != 'spine' and 'anatomy_candidates' in result:
        mapping = {'greater_trochanter': 'greater_trochanter_top',
                   'lesser_trochanter': 'lesser_trochanter', 'ischium': 'ischium_bottom'}
        for landmark in result['anatomy_candidates']['landmarks']:
            if landmark['name'] in mapping:
                overlay[mapping[landmark['name']]] = [(width-1-x if result['region']=='hip_left' else x, y)
                                                     for x,y in landmark['points']]
    return overlay
