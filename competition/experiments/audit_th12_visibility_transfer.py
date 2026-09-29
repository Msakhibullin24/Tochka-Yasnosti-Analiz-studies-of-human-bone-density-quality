"""Check Th12 visibility transfer without inventing DXA reference fractions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import warnings

import joblib
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dxaqc.dicom_io import read_dxa
from experiments.train_th12_visibility import assess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.output.exists():raise ValueError('Choose a new report')
    torch.set_num_threads(4)
    bundle=joblib.load(args.model)
    references=json.loads(args.source_manifest.read_text())
    if references.get('review_package_version')!=3:raise ValueError('Expected source-identity package v3')
    dataset=args.dataset.resolve();cases=[];seen=set()
    for reference in references['cases']:
        if 'Th12' not in reference.get('landmarks', {}):
            continue
        path=(dataset/reference['path_to_file']).resolve()
        if not path.is_relative_to(dataset) or digest(path)!=reference['source_sha256']:
            raise ValueError('Source SHA mismatch')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');image=read_dxa(path)
        if (image.pixel_sha256!=reference['pixel_sha256'] or image.image_uid!=reference['image_uid'] or
            image.pixels.shape!=(reference['height'],reference['width'])):
            raise ValueError('Source pixel or geometry identity mismatch')
        if image.pixel_sha256 in seen:raise ValueError('Duplicate pixel identity in unique-source package')
        seen.add(image.pixel_sha256)
        result=assess(bundle,image.pixels)
        cases.append({'image_uid':image.image_uid,'pixel_sha256':image.pixel_sha256,
                      'source_sha256':reference['source_sha256'],'path_to_file':reference['path_to_file'],
                      **result,'reference':None})
        if len(cases)%25==0:print(f'Checked DXA spine views {len(cases)}',flush=True)
    if not cases:raise ValueError('No spine source cases')
    report={'cases':cases,'images':len(cases),'prediction_counts':dict(Counter(c['candidate_rule'] for c in cases)),
            'accuracy':None,'clinical_validation':False,'release_modified':False,'requirements_complete':False,
            'source_manifest_sha256':digest(args.source_manifest),'model_sha256':digest(args.model),
            'code_sha256':digest(Path(__file__)),
            'limitations':['Original source identities only; no confirmed DXA full-Th12 visibility fractions.',
                           'Intervals were calibrated on ordinary-XR synthetic crops; DXA coverage is unknown.',
                           'Prediction distribution does not establish compliance with the half-body requirement.']}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('images','prediction_counts','accuracy')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','source-manifest','dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
