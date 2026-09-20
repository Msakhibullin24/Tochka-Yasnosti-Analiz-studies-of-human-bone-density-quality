"""Evaluate landmarks/projection against explicitly reviewed independent references.

A template is not a reference. Predictions are deliberately not copied into
reference coordinates. This measures agreement; it cannot authenticate an expert.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

REQUIRED = {'spine': ['Th12', 'iliac_crest_left', 'iliac_crest_right'],
            'hip': ['greater_trochanter', 'lesser_trochanter', 'femoral_neck', 'ischium']}


def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        return [r for r in csv.DictReader(stream) if r.get('processing_status') == 'Success' and not r.get('duplicate_of')]


def template(rows):
    cases=[]
    for row in rows:
        prediction=json.loads(row['anatomy_assessment'])
        group='spine' if any(l['name']=='Th12' for l in prediction['landmarks']) else 'hip'
        cases.append({'image_uid':row['image_uid'], 'path_to_file':row['path_to_file'],
                      'status':'unreviewed','projection':None,
                      'landmarks':{name:{'visible':None,'point':None} for name in REQUIRED[group]}})
    return {'version':1,'coordinate_system':'original_pixel_centres','annotation_source':'',
            'reviewer':'','independent_of_predictions':False,'cases':cases}


def evaluate(rows, reference, tolerance_mm=5.):
    if not math.isfinite(tolerance_mm) or tolerance_mm<=0:
        raise ValueError('tolerance must be positive and finite')
    if (reference.get('version') != 1 or reference.get('coordinate_system')!='original_pixel_centres'
        or not reference.get('reviewer') or not reference.get('annotation_source')
        or reference.get('independent_of_predictions') is not True):
        raise ValueError('independently reviewed reference provenance is required')
    if not reference.get('cases'):
        raise ValueError('reference has no cases')
    indexed={r['image_uid']:r for r in rows}
    if len(indexed)!=len(rows):
        raise ValueError('ambiguous duplicate source UID')
    seen=set();errors={};known={};detected={};false_positives={};unscaled={};projections=[]
    for case in reference['cases']:
        uid=case['image_uid']
        if uid in seen or uid not in indexed or case.get('status')!='confirmed':
            raise ValueError('every reference case must be unique, confirmed and match a prediction')
        seen.add(uid);row=indexed[uid]
        pred=json.loads(row['anatomy_assessment'])
        candidates={l['name']:l['points'] for l in pred['landmarks']}
        group='spine' if 'Th12' in candidates else 'hip'
        if set(case.get('landmarks',{})) != set(REQUIRED[group]):
            raise ValueError('all required landmarks must have explicit visibility labels')
        projection=case.get('projection')
        if projection not in ('frontal','lateral','other'):
            raise ValueError('reference projection must be explicitly labelled')
        estimate=json.loads(row['projection_assessment'])
        projections.append((projection,estimate['value']))
        sx,sy=float(row['pixel_mm_x']),float(row['pixel_mm'])
        scaled=all(math.isfinite(v) and v>0 for v in (sx,sy)) and row['pixel_mm_source'] not in ('device_default','',None)
        for name,label in case['landmarks'].items():
            if label.get('visible') is not True and label.get('visible') is not False:
                raise ValueError('landmark visibility must be explicitly true or false')
            points=candidates.get(name,[])
            if not label['visible']:
                if label.get('point') is not None:
                    raise ValueError('invisible landmark must not have coordinates')
                false_positives[name]=false_positives.get(name,0)+bool(points)
                continue
            point=label.get('point')
            if not isinstance(point,list) or len(point)!=2 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 for v in point):
                raise ValueError('visible landmark requires finite original-image coordinates')
            if point[0] >= int(row['image_width']) or point[1] >= int(row['image_height']):
                raise ValueError('reference point lies outside source image')
            known[name]=known.get(name,0)+1
            detected[name]=detected.get(name,0)+bool(points)
            if points and not scaled:
                unscaled[name]=unscaled.get(name,0)+1
            if points and scaled:
                if len(points)!=1:
                    raise ValueError('named landmark must have exactly one predicted point')
                distance=float(np.linalg.norm((np.asarray(points[0])-point)*[sx,sy]))
                errors.setdefault(name,[]).append(distance)
    metrics={}
    for name in sorted(set(known)|set(false_positives)):
        distances=errors.get(name,[])
        metrics[name]={'visible_references':known.get(name,0),'detected':detected.get(name,0),
                       'false_positive_on_invisible':false_positives.get(name,0),'detected_without_reliable_scale':unscaled.get(name,0),
                       'distances_mm':distances,'median_mm':float(np.median(distances)) if distances else None,
                       'p95_mm':float(np.percentile(distances,95)) if distances else None,
                       'within_tolerance':sum(d<=tolerance_mm for d in distances)}
    return {'evaluated_cases':len(seen),'available_cases':len(rows),'reference_coverage':len(seen)/max(len(rows),1),
            'tolerance_mm':tolerance_mm,'landmarks':metrics,
            'projection':{'correct':sum(a==b for a,b in projections),'abstained':sum(b=='unknown' for _,b in projections),
                          'count':len(projections),'accuracy_including_abstentions':sum(a==b for a,b in projections)/len(projections)},
            'scope':'agreement with supplied reference; reviewer identity and clinical validity not authenticated'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--template',action='store_true')
    mode.add_argument('--reference',type=Path)
    parser.add_argument('--tolerance-mm',type=float,default=5.)
    args=parser.parse_args()
    try:
        rows=read_rows(args.results)
        result=template(rows) if args.template else evaluate(rows,json.loads(args.reference.read_text()),args.tolerance_mm)
        if not args.template:
            result['reference_sha256']=hashlib.sha256(args.reference.read_bytes()).hexdigest()
            result['results_sha256']=hashlib.sha256(args.results.read_bytes()).hexdigest()
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print(f'ANATOMY_VALIDATION_FAILED: {exc}')
        return 2


if __name__=='__main__':
    raise SystemExit(main())
