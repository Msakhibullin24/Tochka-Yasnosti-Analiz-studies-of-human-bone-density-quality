"""Measure the VSD silhouette candidate's input-domain coverage on DXA masks.

There are no reviewed neck-axis labels here: no accuracy score is computed.
The release predictions are only read; this does not change QC decisions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dxaqc.mask_raster import decode_raster
from train_vsd_hip_landmarks import predict_mask


def run(results, model_path, output):
    if output.exists():
        raise ValueError('Choose a new output report')
    bundle=joblib.load(model_path)
    if bundle.get('feature_contract') not in ('binary_mask_bbox_48x48_area_and_aspect_v1','canonical_binary_mask_bbox_48x48_area_and_aspect_v2') or bundle.get('clinical_validation') is not False:
        raise ValueError('Incompatible research silhouette candidate')
    with results.open(encoding='utf-8-sig',newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if r.get('processing_status')=='Success' and not r.get('duplicate_of')]
    cases=[]
    for row in rows:
        anatomy=json.loads(row.get('learned_anatomy') or '{}')
        if anatomy.get('model') != 'hip_masks' and 'femur' not in {r.get('name') for r in anatomy.get('regions',[])}:
            continue
        case={'image_uid':row['image_uid'],'path_to_file':row['path_to_file']}
        masks=[r for r in anatomy.get('regions',[]) if r.get('name')=='femur']
        if len(masks)!=1:
            cases.append({**case,'status':'unavailable','reason':'Missing or ambiguous femur candidate mask'});continue
        shape=(int(row['image_height']),int(row['image_width']))
        try:
            mask=decode_raster(masks[0]['raster'],shape)
            predictions=predict_mask(bundle,mask)
            y,x=np.where(mask);extent=np.array([x.max()-x.min(),y.max()-y.min()])
            inside=((predictions>=0)&(predictions<[shape[1],shape[0]])).all(axis=1)
            if not inside.all():
                raise ValueError('Predicted anatomy falls outside source image')
            span=float(np.linalg.norm(predictions[2]-predictions[3]))
            if span<=1e-8:
                raise ValueError('Degenerate predicted neck axis')
            points={'greater_trochanter':predictions[0].tolist(),'lesser_trochanter':predictions[1].tolist(),
                    'femoral_neck_axis':predictions[2:].tolist()}
            cases.append({**case,'status':'candidate','points':points,'femur_bbox_aspect':float(extent[0]/extent[1]),
                          'mask_source':'existing_unverified_DXA_femur_segmentation','verified':False})
        except ValueError as exc:
            cases.append({**case,'status':'unavailable','reason':str(exc)})
    report={'results_sha256':hashlib.sha256(results.read_bytes()).hexdigest(),
            'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'cases':cases,'hip_mask_cases':len(cases),'candidate_cases':sum(c['status']=='candidate' for c in cases),
            'accuracy':None,'clinical_validation':False,'requirements_complete':False,'release_modified':False,
            'limitations':['Input-domain coverage only; no independent DXA anatomical reference labels.',
                           'Generated points must not be imported into the blind reference editor as expert annotations.',
                           '40x40-derived femur masks differ from detailed 3D silhouettes; candidate extrapolation is unvalidated.']}
    with output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('hip_mask_cases','candidate_cases','accuracy','release_modified')}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('results','model','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.results,args.model,args.output)
