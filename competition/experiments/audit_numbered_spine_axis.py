"""Compare fixed 5-degree spine-axis rules to organizer criterion labels.

The external anatomy model did not train on these organizer spine-axis labels.
This evaluates criterion detection, not vertebral numbering or angle accuracy.
"""
import argparse
import csv
import hashlib
import json
import math
import warnings
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dxaqc.learned_anatomy import numbered_axis
from dxaqc.dicom_io import read_dxa
from source_integrity import inspect_sources


def score(cases,key):
    known=[c for c in cases if c[key] is not None]
    tp=sum(c['axis_violation']==1 and c[key] for c in known)
    fp=sum(c['axis_violation']==0 and c[key] for c in known)
    fn=sum(c['axis_violation']==1 and not c[key] for c in known)
    tn=sum(c['axis_violation']==0 and not c[key] for c in known)
    missing=[c for c in cases if c[key] is None]
    missed=sum(c['axis_violation']==1 for c in missing)
    return {'references':len(cases),'measured':len(known),'unavailable':len(missing),
            'true_positive':tp,'false_positive':fp,'false_negative_measured':fn,'true_negative':tn,
            'positive_unavailable':missed,'sensitivity_including_unavailable':tp/max(1,tp+fn+missed),
            'specificity_on_measured':tn/max(1,tn+fp),'precision':tp/max(1,tp+fp),
            'f1_including_positive_unavailable':2*tp/max(1,2*tp+fp+fn+missed)}


def run(results,labels,dataset,output):
    if output.exists():raise ValueError('Choose a new report')
    dataset=dataset.resolve()
    source_audit=inspect_sources([('organizer',labels,dataset)])
    identities={entry['path']:entry for entry in source_audit['entries']}
    with results.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    indexed={r['path_to_file'].removeprefix('Исследования/'):r for r in rows if r['processing_status']=='Success'}
    if len(indexed)!=sum(r['processing_status']=='Success' for r in rows):raise ValueError('Ambiguous source path')
    with labels.open(newline='') as stream:references=[r for r in csv.DictReader(stream) if r['region']=='spine' and r['spine_axis'] in ('0','1')]
    cases=[]
    for reference in references:
        row=indexed.get(reference['first_source_path'])
        if row is None:raise ValueError('Reference source is missing from inference')
        if (int(reference['rows']),int(reference['columns'])) != (int(row['image_height']),int(row['image_width'])):
            raise ValueError('Image dimensions differ')
        path=(dataset/reference['first_source_path']).resolve()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');image=read_dxa(path)
        if image.image_uid!=row['image_uid'] or image.pixel_sha256!=identities[reference['first_source_path']]['pixel_sha256_current']:
            raise ValueError('DICOM identity or pixels differ from evaluated source')
        features=json.loads(row['measurements'])
        original=features.get('spine_abs_angle_deg')
        original=float(original) if isinstance(original,(int,float)) and math.isfinite(original) else None
        segment=features.get('spine_seg_max_abs_angle_deg')
        segment=float(segment) if isinstance(segment,(int,float)) and math.isfinite(segment) else None
        learned=json.loads(row['learned_anatomy'] or '{}')
        sx,sy=float(row['pixel_mm_x']),float(row['pixel_mm'])
        reliable=np.isfinite([sx,sy]).all() and min(sx,sy)>0 and row.get('pixel_mm_source') not in ('device_default','',None)
        angle=numbered_axis(learned.get('regions',[]),sx,sy) if reliable and learned.get('status')=='evaluated' else None
        case={'source_path':reference['first_source_path'],'study_group':reference['study_key'],
              'image_uid':row['image_uid'],'pixel_sha256':image.pixel_sha256,
              'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'axis_violation':int(reference['spine_axis']),
              'original_angle_deg':original,'numbered_angle_deg':angle,'segment_max_angle_deg':segment,
              'segment_violation':segment>5 if segment is not None else None,
              'original_violation':abs(original)>5 if original is not None else None,
              'numbered_violation':abs(angle)>5 if angle is not None else None,
              'scale_source':row['pixel_mm_source']}
        cases.append(case)
    report={'protocol':'Fixed supplied 5-degree threshold; same expert spine_axis labels; no tuning or new fitting',
            'original':score(cases,'original_violation'),'numbered':score(cases,'numbered_violation'),
            'local_segments':score(cases,'segment_violation'),
            'source_identity_protocol':source_audit['identity_protocol'],
            'historical_label_hash_drift':source_audit['historical_hash_changed'],
            'cases':cases,'results_sha256':hashlib.sha256(results.read_bytes()).hexdigest(),
            'labels_sha256':hashlib.sha256(labels.read_bytes()).hexdigest(),
            'clinical_validation':False,'release_modified':False,'requirements_complete':False,
            'limitations':['Expert binary criterion labels do not provide continuous angle or numbered-vertebra ground truth.',
                           'Candidate segmentation can misnumber vertebrae or substitute vendor ROI for body anatomy.',
                           'This organizer set and previous errors have already been inspected; no new independent cohort.']}
    with output.open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('original','numbered','local_segments')},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('results','labels','dataset','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.results,args.labels,args.dataset,args.output)
