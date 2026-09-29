"""Inspectable anatomical candidates and coverage checks, separate from quality scores.

These deterministic detectors are not expert-validated landmark models. Missing
or ambiguous evidence is never converted to a verified anatomical pass.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import find_peaks

from .geometry import measure_image, spine_centerline
from .source_roi import contains

STABILITY_POINT_MM = 5.0  # technical review distance, not validated landmark accuracy
STABILITY_AXIS_DEG = 1.0
STABILITY_TROCH_MM = 5.0


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
        purpose = roi.get('purpose', 'unknown')
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
        relationships = []
        if purpose == 'femoral_neck' and region != 'spine':
            for name, expected in (('femoral_neck', True), ('greater_trochanter', False)):
                relationships.append({'landmark': name, 'expected_inside': expected,
                                      'candidate_inside': covered.get(name), 'status': 'undetermined',
                                      'reason': 'Подтвердите ориентир; точка не заменяет контур структуры.'})
        checks.append({'roi_id':roi['id'], 'purpose': purpose, 'status':'needs_review', 'anatomical_validity':'undetermined',
                       'anatomical_relationships': relationships,
                       'candidate_landmarks_inside':covered,
                       'margins_mm':{k:round(v,3) for k,v in margins.items()} if region!='spine' else None,
                       'margin_status':('pass' if all(margins[k]>=v-1e-8 for k,v in {'top':30,'bottom':30,'side':20}.items()) else 'fail') if known_scale and region!='spine' and purpose=='scan_coverage' else 'not_applicable' if purpose!='scan_coverage' else 'unavailable',
                       'scale_source':img.pixel_mm_source,'edge':edge,
                       'reason':'Назначение исходной ROI и анатомические ориентиры требуют подтверждения; попадание кандидатов не доказывает корректность разметки.'})
    return {'status':source['status'],'rois':source['rois'],'issues':source['issues'],'checks':checks}


def compare_hip_stability(base, altered, base_features, altered_features, pixel_mm_y, pixel_mm_x):
    """Suppress only display candidates that fail a deterministic repeatability check."""
    shifted = {item['name']: item['points'] for item in altered['landmarks']}
    unstable = []
    for item in base['landmarks']:
        points, other = item['points'], shifted.get(item['name'], [])
        distance = (float(np.linalg.norm((np.asarray(points[0]) - other[0]) *
                                         [pixel_mm_x, pixel_mm_y])) if len(points) == len(other) == 1 else None)
        changed = bool(points) != bool(other) or (distance is not None and distance > STABILITY_POINT_MM)
        item['stability'] = {'status': 'unstable' if changed else 'repeatable' if points else 'unavailable',
                             'shift_mm': round(distance, 3) if distance is not None else None,
                             'test': 'brightness_1.1x_minus_3', 'limit_mm': STABILITY_POINT_MM}
        if changed:
            unstable.append(item['name'])
            item['points'] = []
            item['status'] = 'unstable'
    changes = {}
    for name, limit in (('shaft_abs_angle_deg', STABILITY_AXIS_DEG),
                        ('lesser_troch_protrusion_mm', STABILITY_TROCH_MM)):
        first, second = base_features.get(name), altered_features.get(name)
        change = (abs(float(first-second)) if first is not None and second is not None
                  and np.isfinite(first) and np.isfinite(second) else None)
        changes[name] = {'delta': round(change, 3) if change is not None else None,
                         'limit': limit, 'unstable': change is None or change > limit}
    return {'status': 'needs_review' if unstable or any(x['unstable'] for x in changes.values())
            else 'repeatable_under_test', 'test': 'brightness_1.1x_minus_3',
            'unstable_landmarks': unstable, 'measurements': changes,
            'clinical_validation': False}


def image_assessment(img, result):
    """Shared single-image and batch contract. Cache only pixel-derived evidence."""
    if 'anatomy_candidates' not in result:
        result['anatomy_candidates'] = detect_landmarks(img.pixels, result['region'], result['overlay'], img.pixel_mm, img.pixel_mm_x or img.pixel_mm)
        if result['region'] != 'spine':
            if getattr(img, 'pixel_mm_source', '') in ('device_default', '', None):
                stability = {'status': 'unavailable', 'reason': 'pixel_scale_uncertain',
                             'clinical_validation': False}
            else:
                bright = np.clip(img.pixels.astype(np.float32) * 1.1 - 3, 0, 255).astype(np.uint8)
                canonical = np.ascontiguousarray(bright[:, ::-1]) if result['region'] == 'hip_left' else bright
                shifted_measurement = measure_image(canonical, result['region'], img.pixel_mm, img.pixel_mm_x)
                shifted = detect_landmarks(bright, result['region'], shifted_measurement.overlay,
                                           img.pixel_mm, img.pixel_mm_x or img.pixel_mm)
                stability = compare_hip_stability(result['anatomy_candidates'], shifted, result['features'],
                                                  shifted_measurement.features, img.pixel_mm,
                                                  img.pixel_mm_x or img.pixel_mm)
            stability['scope'] = 'heuristic_geometry_only'
            result['anatomy_candidates']['stability'] = stability
            if stability['status'] == 'needs_review' and 'hip_geometry_unstable' not in result['review_reasons']:
                result['review_reasons'].append('hip_geometry_unstable')
        learned = result.get('learned_anatomy', {})
        if learned.get('status') == 'evaluated':
            by_name = {r['name']: r for r in learned['regions']}
            if result['region'] == 'spine':
                for item in result['anatomy_candidates']['landmarks']:
                    if item['name'] == 'Th12':
                        item.update(points=[by_name['Th12']['center']] if 'Th12' in by_name else [],
                                    method='learned_named_vertebral_mask',
                                    status='candidate' if 'Th12' in by_name else 'not_localized')
                result['anatomy_candidates']['landmarks'].append({
                    'name': 'numbered_vertebral_centers', 'points': [by_name[f'L{i}']['center'] for i in range(1,5) if f'L{i}' in by_name],
                    'method': 'learned_named_mask_centers', 'status': 'candidate', 'verified': False})
            elif 'femoral_neck_roi' in by_name:
                for item in result['anatomy_candidates']['landmarks']:
                    if item['name'] == 'femoral_neck':
                        item.update(points=[by_name['femoral_neck_roi']['center']], status='candidate',
                                    method='learned_vendor_neck_roi_center',
                                    stability={'status': 'not_evaluated', 'reason': 'learned_mask_stability_not_tested'})
            result['anatomy_candidates']['learned_model_sha256'] = learned['model_sha256']
        result['projection_pixels'] = assess_projection(img.pixels, result['region'], result['anatomy_candidates'])
    projection = {**result['projection_pixels'], 'declared_view_position': img.view_position}
    if result.get('learned_projection') is not None:
        projection = {**result['learned_projection'], 'declared_view_position': img.view_position,
                      'heuristic_candidate': result['projection_pixels']}
    from .requirements import image_checks
    source_roi = evaluate_source_roi(img.source_roi, result['anatomy_candidates'], img, result['region'])
    # A labelled scan-coverage ROI and explicit calibration supply an exact
    # geometric check. This rule does not depend on guessed bone landmarks and
    # must apply to this DICOM's ROI, never to the shared pixel-cache result.
    apply_source_roi_findings(result, source_roi)
    if result.get('learned_anatomy', {}).get('status') == 'evaluated':
        from .anatomical_roi import compare_source_rois
        source_roi['named_mask_comparison'] = compare_source_rois(
            img.source_roi, result['learned_anatomy']['regions'], img.pixels.shape)
    if result['region'] != 'spine':
        from .neck_roi_geometry import assess_neck_rois
        source_roi['neck_geometry'] = assess_neck_rois(
            img.source_roi, result.get('learned_anatomy', {}).get('regions', []), img)
    checks = image_checks(img, result, projection, source_roi)
    assessment = {'projection': projection, 'anatomy': result['anatomy_candidates'],
            'source_roi': source_roi, 'checks': checks,
            'complete': all(c['status'] in ('evaluated', 'not_applicable') for c in checks), 'version': '4'}
    if 'specialist_qc' in result:
        assessment['specialist_qc'] = result['specialist_qc']
    if 'specialist_outputs' in result:
        assessment['specialist_outputs'] = result['specialist_outputs']
    if 'learned_anatomy' in result:
        assessment['learned_anatomy'] = result['learned_anatomy']
    return assessment


def apply_source_roi_findings(result, source_roi):
    """Integrate measured source-ROI failures without retyping absent/neck ROIs."""
    if result['region'] == 'spine':
        return
    failed = [check for check in source_roi['checks']
              if check.get('purpose') == 'scan_coverage' and check.get('margin_status') == 'fail']
    if not failed:
        return
    code = 'hip_roi_coverage'
    if code not in result['violations']:
        result['violations'] = [*result['violations'], code]
    result['quality'] = 1
    result['violation_type_status'] = 'identified'
    result['criterion_states'] = {**result['criterion_states'], code: {
        'status': 'fail', 'basis': 'explicit_source_roi_scan_margins',
        'limits_mm': {'top': 30., 'bottom': 30., 'side': 20.},
        'roi_ids': [check['roi_id'] for check in failed],
        'clinical_validation': False}}
    result['review_reasons'] = [reason for reason in result['review_reasons']
                                if reason != 'violation_type_undetermined']
    result['decision_reason'] = 'source_roi_scan_margin_failure'
    if not result['decision_version'].endswith('+source-roi-1'):
        result['decision_version'] += '+source-roi-1'


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
