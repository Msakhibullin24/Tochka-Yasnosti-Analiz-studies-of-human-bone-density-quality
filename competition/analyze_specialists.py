"""Run available specialist models locally with explicit protocol and output semantics.

COCO boxes remain COCO boxes. Raw DXA-to-3D output is not mapped to vertebral IDs.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch
from dxaqc.dicom_io import read_any
from dxaqc.specialist_catalog import inventory
from dxaqc.specialist_qc import load_specialist, image_tensor


def run(args):
    if args.output.exists():raise ValueError('Choose a new output directory')
    state=inventory(args.assets)
    entries={c['id']:c for c in state['capabilities']}
    image=read_any(args.input)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    report={'protocol':args.protocol,'clinical_validation':False,'affects_decision':False,
            'image_shape':list(image.pixels.shape),'models':{},'inventory':state}
    if args.protocol in ('spine','hip_left','hip_right'):
        if args.qc:
            report['models']['trained_qc']=load_specialist(args.qc).predict(image.pixels,args.protocol)
        else:
            report['models']['trained_qc']={'status':'not_configured'}
    if entries['yolo26x']['artifacts_verified']:
        os.environ.setdefault('YOLO_CONFIG_DIR',str((args.assets/'ultralytics-config').resolve()))
        from ultralytics import YOLO
        for name in ('yolo26x','yolo26x-seg'):
            model=YOLO(str((args.assets/'weights'/f'{name}.pt').resolve()))
            result=model.predict(np.repeat(image.pixels[:,:,None],3,axis=2),device='cpu',imgsz=640,verbose=False)[0]
            report['models'][name]={'status':'generic_coco_inference','dxa_artifact_model':False,
                'objects':[{'class':model.names[int(box.cls.item())],'confidence':float(box.conf.item()),
                            'xyxy':box.xyxy[0].tolist()} for box in result.boxes]}
            if result.masks is not None:
                np.savez_compressed(args.output/f'{name}-masks.npz',masks=result.masks.data.cpu().numpy())
                report['models'][name]['mask_coordinates']='Ultralytics inference raster; use result.masks.xy for original polygons'
                report['models'][name]['polygons_xy']=[poly.tolist() for poly in result.masks.xy]
    if args.protocol=='whole-body-air-ratio':
        if entries['totalbody_105']['artifacts_verified']:
            cv2.imwrite(str(args.output/'input.png'),image.pixels)
            root=Path(__file__).resolve().parents[1]
            completed=subprocess.run([str(root/'backend/.venv/bin/python'),str(root/'backend/scripts/predict_totalbody.py'),
                                      '--image',str(args.output/'input.png'),'--output',str(args.output/'totalbody.json')],
                                      capture_output=True,text=True,timeout=180)
            (args.output/'totalbody.log').write_text(completed.stdout+completed.stderr)
            report['models']['totalbody_105']=(json.loads((args.output/'totalbody.json').read_text()) if completed.returncode==0
                                              else {'status':'runtime_error','log':'totalbody.log'})
        if entries['dxa_to_3d']['artifacts_verified']:
            from check_specialists import shape_model
            weights=torch.load(args.assets/'weights/dxa_to_3d.pt',map_location='cpu',weights_only=True)
            model=shape_model(weights['linear.weight'].shape[0]);model.load_state_dict(weights,strict=True)
            with torch.inference_mode():vector=model(image_tensor(image.pixels,224)[None]).numpy()
            if not np.isfinite(vector).all():raise ValueError('Non-finite shape output')
            np.save(args.output/'dxa-to-3d-raw.npy',vector)
            report['models']['dxa_to_3d']={'status':'raw_research_output','shape':list(vector.shape),
                'limitation':'Input transform and curve order unvalidated; no numbered vertebrae, physical angles or 3D reconstruction claimed.'}
    else:
        for name in ('totalbody_105','dxa_to_3d'):
            report['models'][name]={'status':'not_applicable','reason':'Whole-body source protocol required'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--protocol',choices=['spine','hip_left','hip_right','whole-body-air-ratio'],required=True)
    p.add_argument('--qc',type=Path)
    p.add_argument('--assets',type=Path,default=Path('data/specialists'))
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
