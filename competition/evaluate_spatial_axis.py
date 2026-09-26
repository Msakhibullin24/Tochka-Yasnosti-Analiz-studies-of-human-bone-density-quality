"""Compare physical axis candidates from numbered centers; do not alter verdicts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dxaqc.dicom_io import read_dxa
from evaluate_organizer_dataset import sha256, binary_metrics
from predict_spatial_anatomy import load, predict


def axis_angle(points, pixel_mm_x, pixel_mm_y):
    points = np.asarray(points, dtype=float)
    if (points.shape != (4, 2) or not np.isfinite(points).all()
            or not np.isfinite([pixel_mm_x, pixel_mm_y]).all()
            or min(pixel_mm_x, pixel_mm_y) <= 0 or not (np.diff(points[:, 1]) > 0).all()):
        raise ValueError('Need ordered L1-L4 centers and positive physical spacing')
    physical = points*np.array([pixel_mm_x, pixel_mm_y])
    centered = physical-physical.mean(axis=0)
    slope = (centered[:, 0]*centered[:, 1]).sum()/(centered[:, 1]**2).sum()
    return float(np.degrees(np.arctan(slope)))


def run(dataset, labels, model_path, dxa_oof, public_root, output):
    if output.exists():
        raise ValueError('Choose a new output path')
    torch.set_num_threads(4)
    model = load(model_path)
    public = json.loads(dxa_oof.read_text())
    transfer, public_cases = [], []
    for row in public['cases']:
        source = public_root/'Annotation'/row['patient_id']/'images/spine_image.dcm'
        if sha256(source) != row['source_sha256']:
            raise ValueError('Public reference changed')
        image = read_dxa(source)
        ref = axis_angle(row['reference'], image.pixel_mm_x or image.pixel_mm, image.pixel_mm)
        try:
            value = axis_angle(row['prediction'], image.pixel_mm_x or image.pixel_mm, image.pixel_mm)
        except ValueError:
            value = None
        public_cases.append({'source_sha256':sha256(source),'reference_roi_axis_deg':ref,
                             'oof_candidate_axis_deg':value,'clinical_validation':False})
    with labels.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['region']=='spine' and r['quality_class'] in ('0','1')]
    for row in rows:
        source = dataset/row['first_source_path'];image = read_dxa(source)
        centers = predict(model, image.pixels)[:4]
        try:
            angle = axis_angle(centers, image.pixel_mm_x or image.pixel_mm, image.pixel_mm)
        except ValueError:
            angle = None
        transfer.append({'source_path':row['first_source_path'],'source_sha256':sha256(source),
                         'pixel_sha256':image.pixel_sha256,'reference_axis_label':row['spine_axis'],
                         'candidate_axis_deg':angle,'numbered_centers':centers.tolist(),
                         'original_verdict_changed':False})
    known = [r for r in transfer if r['reference_axis_label'] in ('0','1') and r['candidate_axis_deg'] is not None]
    errors = [abs(r['oof_candidate_axis_deg']-r['reference_roi_axis_deg']) for r in public_cases
              if r['oof_candidate_axis_deg'] is not None]
    report = {'clinical_validation':False,'release_modified':False,'model_sha256':sha256(model_path),
              'code_sha256':sha256(Path(__file__)),'axis_limit_deg':5.,
              'protocol':'Fixed 5-degree rule; L1-L4 physical x/y spacing; DXA ROI model never trained on organizer images',
              'public_oof':{'patients':len(public_cases),'undetermined':sum(r['oof_candidate_axis_deg'] is None for r in public_cases),
                            'median_absolute_error_deg':float(np.median(errors)) if errors else None,
                            'max_absolute_error_deg':max(errors,default=None),'cases':public_cases},
              'organizer_axis_metrics':binary_metrics(np.array([int(r['reference_axis_label']) for r in known]),
                                                     np.array([int(abs(r['candidate_axis_deg'])>5.) for r in known])),
              'organizer_undetermined':sum(r['candidate_axis_deg'] is None for r in transfer),'cases':transfer,
              'limitations':['An estimated angle is not clinical verification of the predicted vertebral identities.',
                              'Public ROI center axis is a weak surrogate, not anatomical axis ground truth.',
                              'Organizer criterion labels are existing references; no new physician annotation or label correction.',
                              'Organizers provide GE spacing via protocol fallback; coordinate uncertainty remains.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','limitations','public_oof')},indent=2))
    print('Public OOF angle error',report['public_oof']['median_absolute_error_deg'],flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','labels','model','dxa-oof','public-root','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();run(a.dataset,a.labels,a.model,a.dxa_oof,a.public_root,a.output)
