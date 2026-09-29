"""Generate partial-Th12 views from published full-body quadrilaterals only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import numpy as np
from scipy.io import loadmat

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anatomy_training_data import load_record,lumbar_crop_bounds
from spinal_ai_annotations import quad_polygon
from amodal_visibility import projected_visibility

FRACTIONS=(0.,.25,.5,.75,1.)


def run(inventory_path,output):
    if output.exists():raise ValueError('Choose a new output directory')
    source=json.loads(inventory_path.read_text())['rows']
    parents=[r for r in source if r['source']=='aasce']
    if len({r['group'] for r in parents})!=len(parents):
        raise ValueError('Use an inventory with one AASCE view per original parent')
    if not parents or {r['split'] for r in parents}!={'train','test'}:
        raise ValueError('Both original parent train/test splits required')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.th12-amodal-',dir=output.parent))
    cases=[];total=0
    try:
        (stage/'images').mkdir()
        for i,row in enumerate(parents):
            original={k:v for k,v in row.items() if k not in ('crop_xyxy','view')}
            pixels,target=load_record(original)
            points=loadmat(row['annotation'])['p2'].reshape(17,4,2)[11]
            full=np.asarray(quad_polygon(points.ravel(),pixels.shape[1],pixels.shape[0])).reshape(4,2)
            x0,_,x1,y1=lumbar_crop_bounds(target)
            top,bottom=float(full[:,1].min()),float(full[:,1].max())
            for nominal in FRACTIONS:
                y0=max(0,int(np.floor(top))-2) if nominal==1 else int(np.ceil(bottom+.5)) if nominal==0 else int(np.floor(bottom-(bottom-top)*nominal))
                if y0>=y1:raise ValueError('Partial-body view has no lumbar field')
                crop=np.ascontiguousarray(pixels[y0:y1,x0:x1])
                translated=full-[x0,y0]
                visibility=projected_visibility(translated,crop.shape)
                name=f'images/{i:04d}-{int(nominal*100):03d}.png'
                ok,encoded=cv2.imencode('.png',crop)
                if not ok:raise ValueError('Cannot encode source crop')
                raw=encoded.tobytes();total+=len(raw)
                if total>1_000_000_000:raise ValueError('Derived raster budget exceeded')
                (stage/name).write_bytes(raw)
                cases.append({'image':name,'image_sha256':hashlib.sha256(raw).hexdigest(),
                              'parent_group':row['group'],'split':row['split'],
                              'parent_image_sha256':row['image_sha256'],'annotation_sha256':row['annotation_sha256'],
                              'nominal_vertical_fraction':nominal,'crop_xyxy':[x0,y0,x1,y1],
                              'full_body_quad':translated.tolist(),'width':crop.shape[1],'height':crop.shape[0],
                              'reference':visibility,'reference_origin':'published full Th12 quadrilateral transformed to a derived crop'})
            if (i+1)%50==0:print(f'Prepared Th12 parent views {i+1}/{len(parents)}',flush=True)
        if {c['parent_group'] for c in cases if c['split']=='train'} & {c['parent_group'] for c in cases if c['split']=='test'}:
            raise ValueError('Derived-view parent leakage')
        report={'version':1,'cases':cases,'parent_images':len(parents),'derived_images':len(cases),'bytes':total,
                'parent_inventory_sha256':hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'geometry_code_sha256':hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('amodal_visibility.py').read_bytes()).hexdigest(),
                'clinical_validation':False,'requirements_complete':False,'release_modified':False,
                'limitations':['Derived crops of ordinary radiographs, not independently labelled DXA visibility.',
                               'Publisher quadrilaterals approximate projected bodies, not complete 3D bone contours.',
                               'Area and vertical-height fractions are reported separately; no medical equivalence is asserted.',
                               'Parent identity is image-based; source patient identity is unavailable.',
                               'All variants inherit the previously examined source split; no new independent cohort.']}
        (stage/'dataset.json').write_text(json.dumps(report,indent=2)+'\n')
        stage.rename(output)
        print(json.dumps({k:report[k] for k in ('parent_images','derived_images','bytes')},indent=2))
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.inventory,args.output)
