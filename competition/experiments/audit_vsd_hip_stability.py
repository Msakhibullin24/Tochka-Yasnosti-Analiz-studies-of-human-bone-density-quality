"""Reflection consistency of the research VSD localizer on existing DXA masks.

Consistency does not measure agreement with anatomy or clinical QC accuracy.
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
from train_vsd_hip_landmarks import predict_mask, ORDER


def run(results, model_path, output):
    if output.exists(): raise ValueError('Choose a new output report')
    bundle=joblib.load(model_path)
    def predict(mask):
        return predict_mask(bundle,mask)
    with results.open(encoding='utf-8-sig',newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if r.get('processing_status')=='Success' and not r.get('duplicate_of')]
    cases=[]
    for row in rows:
        anatomy=json.loads(row.get('learned_anatomy') or '{}')
        masks=[r for r in anatomy.get('regions',[]) if r.get('name')=='femur']
        if len(masks)!=1:continue
        shape=(int(row['image_height']),int(row['image_width']))
        case={'image_uid':row['image_uid'],'path_to_file':row['path_to_file']}
        try:
            mask=decode_raster(masks[0]['raster'],shape)
            first=predict(mask);mirrored=predict(mask[:,::-1].copy());mirrored[:,0]=shape[1]-1-mirrored[:,0]
            sx,sy=float(row['pixel_mm_x']),float(row['pixel_mm'])
            scaled=np.isfinite([sx,sy]).all() and min(sx,sy)>0 and row.get('pixel_mm_source') not in ('device_default','',None)
            delta=first-mirrored
            drift=np.linalg.norm(delta*[sx,sy],axis=1) if scaled else None
            a,b=(first[2]-first[3])*[sx,sy],(mirrored[2]-mirrored[3])*[sx,sy]
            denom=np.linalg.norm(a)*np.linalg.norm(b)
            angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(a,b))/denom,0,1)))) if scaled and denom>0 else None
            cases.append({**case,'status':'measured','landmark_drift_pixels':dict(zip(ORDER,np.linalg.norm(delta,axis=1).tolist())),
                          'landmark_drift_mm':dict(zip(ORDER,drift.tolist())) if scaled else None,
                          'neck_axis_drift_deg':angle,'reliable_scale':bool(scaled)})
        except ValueError as exc:cases.append({**case,'status':'unavailable','reason':str(exc)})
    measured=[c for c in cases if c['status']=='measured'];scaled=[c for c in measured if c['reliable_scale']]
    values=[c['neck_axis_drift_deg'] for c in scaled if c['neck_axis_drift_deg'] is not None]
    report={'protocol':'Reflect each unchanged binary femur mask horizontally; infer frozen candidate; map points back; compare to original inference',
            'cases':cases,'measured_cases':len(measured),'reliably_scaled_cases':len(scaled),
            'neck_axis_mean_drift_deg':float(np.mean(values)) if values else None,
            'neck_axis_p95_drift_deg':float(np.percentile(values,95)) if values else None,
            'landmark_mean_drift_mm':{name:float(np.mean([c['landmark_drift_mm'][name] for c in scaled])) for name in ORDER} if scaled else {},
            'results_sha256':hashlib.sha256(results.read_bytes()).hexdigest(),
            'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'accuracy':None,'clinical_validation':False,'release_modified':False,'requirements_complete':False,
            'scope':'Reflection consistency only; no independent anatomical reference; no invented acceptance threshold'}
    with output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases',)},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('results','model','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.results,args.model,args.output)
