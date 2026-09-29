"""Separate named-level errors from localization on held-out published references."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anatomy_training_data import load_record
from dxaqc.learned_anatomy import LearnedAnatomy, CLASSES


def largest_components(labels,known):
    masks={}
    for level in range(1,len(CLASSES)):
        mask=((labels==level)&known).astype(np.uint8)
        count,components,stats,_=cv2.connectedComponentsWithStats(mask,8)
        if count>1:
            biggest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
            masks[level]=components==biggest
    return masks


def numbering_metrics(predicted,reference,source_shape,source_levels=None):
    """Object IoU50 is a reporting convention, never a clinical acceptance rule."""
    predicted,reference=np.asarray(predicted),np.asarray(reference)
    if predicted.shape!=reference.shape or predicted.ndim!=2:raise ValueError('Compatible label grids required')
    known=reference!=-100
    if not np.isin(reference[known],np.arange(len(CLASSES))).all() or not np.isin(predicted,np.arange(len(CLASSES))).all():
        raise ValueError('Unknown named label')
    levels=list(source_levels) if source_levels is not None else [int(v) for v in np.unique(reference[known]) if v>0]
    if len(set(levels))!=len(levels) or any(v not in range(1,len(CLASSES)) for v in levels):
        raise ValueError('Invalid original reference levels')
    references={int(v):reference==v for v in levels}
    predictions=largest_components(predicted,known)
    a,b=list(references),list(predictions)
    overlaps=np.zeros((len(a),len(b)))
    for i,truth in enumerate(a):
        for j,guess in enumerate(b):
            overlaps[i,j]=float((references[truth]&predictions[guess]).sum()/(references[truth]|predictions[guess]).sum())
    assigned={}
    if overlaps.size:
        rows,cols=linear_sum_assignment(-overlaps)
        assigned={a[i]:(b[j],float(overlaps[i,j])) for i,j in zip(rows,cols) if overlaps[i,j]>=.5}
    sx,sy=source_shape[1]/reference.shape[1],source_shape[0]/reference.shape[0]
    cases=[]
    for name,mask in references.items():
        match=assigned.get(name)
        same=predictions.get(name)
        boundary=None
        if same is not None and mask.any():
            # Distances use source pixels, accounting for both resize dimensions.
            actual_edge=mask & ~binary_erosion(mask)
            predicted_edge=same & ~binary_erosion(same)
            actual_distance=distance_transform_edt(~actual_edge,sampling=(sy,sx))
            predicted_distance=distance_transform_edt(~predicted_edge,sampling=(sy,sx))
            distances=np.r_[actual_distance[predicted_edge],predicted_distance[actual_edge]]
            boundary=float(np.percentile(distances,95))
        cases.append({'reference_name':CLASSES[name], 'reference_resolved_on_grid':bool(mask.any()), 'localized_iou50':match is not None,
                      'matched_prediction_name':CLASSES[match[0]] if match else None,
                      'matched_iou':match[1] if match else None,
                      'numbered_correctly_iou50':bool(match and match[0]==name),
                      'same_name_boundary_hd95_source_pixels':boundary})
    return {'references':len(a),'predictions_in_known_area':len(b),'localized_iou50':len(assigned),
            'correct_names_iou50':sum(c['numbered_correctly_iou50'] for c in cases),
            'false_predictions_iou50':len(b)-len(assigned),'cases':cases}


def summarize(cases,model):
    values=[c[model] for c in cases]
    refs=sum(v['references'] for v in values);located=sum(v['localized_iou50'] for v in values)
    correct=sum(v['correct_names_iou50'] for v in values)
    boundaries=[r['same_name_boundary_hd95_source_pixels'] for v in values for r in v['cases'] if r['same_name_boundary_hd95_source_pixels'] is not None]
    confusion=Counter(f"{r['reference_name']}->{r['matched_prediction_name']}" for v in values for r in v['cases'] if r['localized_iou50'])
    return {'images':len(cases),'reference_bodies':refs,'localized_iou50':located,
            'localized_fraction':located/refs if refs else None,'correct_names_iou50':correct,
            'correct_named_fraction_including_misses':correct/refs if refs else None,
            'correct_names_among_localized':correct/located if located else None,
            'false_predictions_iou50':sum(v['false_predictions_iou50'] for v in values),
            'references_lost_by_grid_quantization':sum(not r['reference_resolved_on_grid'] for v in values for r in v['cases']),
            'named_confusion':dict(sorted(confusion.items())),
            'boundary_hd95_mean_source_pixels_on_present_names':float(np.mean(boundaries)) if boundaries else None,
            'boundary_measured_bodies':len(boundaries),'boundary_unavailable_bodies':refs-len(boundaries)}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.output.exists():raise ValueError('Choose a new report')
    torch.set_num_threads(4)
    inventory_path=args.training/'inventory.json'
    rows=json.loads(inventory_path.read_text())['rows']
    if {r['group'] for r in rows if r['split']=='test'} & {r['group'] for r in rows if r['split']=='train'}:
        raise ValueError('Parent leakage')
    features=np.load(args.training/'features.npy',mmap_mode='r')
    if features.shape!=(len(rows),896,40,40):raise ValueError('Unexpected feature cache')
    models={'deployed':LearnedAnatomy(args.deployed),'candidate':LearnedAnatomy(args.training/'anatomy_masks.pt')}
    cases=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            if row['split']!='test':continue
            pixels,reference=load_record(row)
            original_levels=[int(v) for v in np.unique(reference) if v>0]
            reference=cv2.resize(reference.astype(np.float32),(40,40),interpolation=cv2.INTER_NEAREST).astype(np.int16)
            x=torch.from_numpy(np.array(features[i],dtype=np.float32))[None]
            case={k:row[k] for k in ('source','view','group','image_sha256','annotation_sha256','target_semantics')}
            for name,model in models.items():
                predicted=model.model(x)[0].argmax(0).numpy()
                case[name]=numbering_metrics(predicted,reference,pixels.shape,original_levels)
            cases.append(case)
    summary={}
    for source,view in sorted({(c['source'],c['view']) for c in cases}):
        selected=[c for c in cases if (c['source'],c['view'])==(source,view)]
        summary[source+'/'+view]={name:summarize(selected,name) for name in models}
    report={'protocol':'Frozen heads and original parent holdout; match largest components independent of names with one-to-one maximum IoU; report at IoU>=0.5.',
            'summary':summary,'cases':cases,'model_sha256':{k:m.sha256 for k,m in models.items()},
            'inventory_sha256':digest(inventory_path),'feature_cache_sha256':digest(args.training/'features.npy'),
            'code_sha256':digest(Path(__file__)),'clinical_validation':False,'release_modified':False,
            'requirements_complete':False,
            'limitations':['Previously examined source holdout: exploratory audit, not new independent validation.',
                           'IoU50 is a localization reporting convention, not a threshold provided by the medical specification.',
                           'References and predictions evaluated on 40x40 grid; source-pixel distances include quantization. Original bodies lost on resize remain in denominators and are counted separately.',
                           'Quadrilateral/vendor-ROI boundaries are not independently annotated intervertebral disc lines.',
                           'Unknown Hologic anatomy is masked; false detections outside labelled area are not assessed.',
                           'No GE anatomical numbering references or physical-mm boundary accuracy are supplied.']}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:{m:{k:v for k,v in values.items() if k!='named_confusion'} for m,values in value.items()} for key,value in summary.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','deployed','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
