"""Subject-disjoint silhouette landmark feasibility experiment; not a DXA model."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

ORDER = ('greater_trochanter', 'lesser_trochanter', 'head_center', 'neck_isthmus_center')


def features(mask):
    mask = np.asarray(mask)>0
    y,x = np.where(mask)
    if len(x)<100:
        raise ValueError('Nonempty femur silhouette required')
    origin = np.array([x.min(),y.min()],float)
    extent = np.array([x.max()-x.min(),y.max()-y.min()],float)
    if min(extent)<5:
        raise ValueError('Degenerate femur silhouette')
    crop = mask[y.min():y.max()+1,x.min():x.max()+1].astype(np.float32)
    normalized = cv2.resize(crop,(48,48),interpolation=cv2.INTER_AREA)
    # The bounding box's aspect ratio remains visible after normalization.
    vector = np.r_[normalized.ravel(),extent[0]/extent[1]]
    return vector, origin, extent


def canonical_features(mask):
    mask=np.asarray(mask)>0
    y,x=np.where(mask)
    if len(x)<100:
        raise ValueError('Nonempty femur silhouette required')
    middle=(y.min()+y.max())/2
    top=x[y<=middle];bottom=x[y>middle]
    if not len(top) or not len(bottom):
        raise ValueError('Cannot determine silhouette orientation')
    direction=float(top.mean()-bottom.mean())
    if abs(direction)<=1e-8:
        raise ValueError('Silhouette orientation is ambiguous')
    reflected=direction>0
    vector,origin,extent=features(mask[:,::-1] if reflected else mask)
    return vector,origin,extent,reflected


def predict_mask(bundle, mask):
    contract=bundle.get('feature_contract')
    if contract=='canonical_binary_mask_bbox_48x48_area_and_aspect_v2':
        vector,origin,extent,reflected=canonical_features(mask)
    elif contract=='binary_mask_bbox_48x48_area_and_aspect_v1':
        vector,origin,extent=features(mask);reflected=False
    else:
        raise ValueError('Incompatible silhouette feature contract')
    points=bundle['model'].predict(vector[None]).reshape(4,2)*extent+origin
    if reflected:points[:,0]=mask.shape[1]-1-points[:,0]
    return points


def measures(prediction, truth, selected, rows):
    distance = np.linalg.norm((prediction-truth)*.5,axis=2)
    a,b = prediction[:,2]-prediction[:,3], truth[:,2]-truth[:,3]
    denom = np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    good = denom>1e-8
    errors = np.degrees(np.arccos(np.clip(np.abs((a[good]*b[good]).sum(1))/denom[good],0,1)))
    return {'images':len(selected),'subjects':len({rows[i]['subject'] for i in selected}),
            'mean_point_error_mm':{name:float(distance[:,i].mean()) for i,name in enumerate(ORDER)},
            'mean_neck_axis_error_deg':float(errors.mean()) if len(errors) else None,
            'p95_neck_axis_error_deg':float(np.percentile(errors,95)) if len(errors) else None,
            'degenerate_axis_predictions':int((~good).sum())}


def run(dataset, output, canonical=False):
    if output.exists():
        raise ValueError('Choose a new experiment output')
    metadata=json.loads((dataset/'dataset.json').read_text())
    rows=metadata['records']
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    if not len(train) or not len(test) or {rows[i]['subject'] for i in train}&{rows[i]['subject'] for i in test}:
        raise ValueError('Subject-disjoint train/test required')
    xs,ys,origins,extents,truths,hashes,reflections=[],[],[],[],[],[],[]
    for row in rows:
        path=(dataset/row['image']).resolve()
        if not path.is_relative_to(dataset.resolve()):
            raise ValueError('Image path escapes dataset')
        image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError('Unreadable silhouette')
        if canonical: x,origin,extent,reflected=canonical_features(image)
        else: x,origin,extent=features(image);reflected=False
        reflections.append(reflected)
        points=np.vstack([row['landmarks']['greater_trochanter'],row['landmarks']['lesser_trochanter'],row['landmarks']['femoral_neck_axis']])
        canonical_points=points.copy()
        if reflected:canonical_points[:,0]=image.shape[1]-1-canonical_points[:,0]
        xs.append(x);ys.append(((canonical_points-origin)/extent).ravel());origins.append(origin);extents.append(extent);truths.append(points)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    xs,ys,origins,extents,truths=map(np.asarray,(xs,ys,origins,extents,truths))
    # Fixed protocol before inspecting the held-out subjects: no test-set tuning.
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=2,max_features=1.,random_state=17,n_jobs=4)
    model.fit(xs[train],ys[train])
    prediction=model.predict(xs[test]).reshape(-1,4,2)*extents[test,None,:]+origins[test,None,:]
    prior=np.broadcast_to(ys[train].mean(0).reshape(4,2),(len(test),4,2))*extents[test,None,:]+origins[test,None,:]
    for j,i in enumerate(test):
        if reflections[i]:
            prediction[j,:,0]=320-1-prediction[j,:,0]
            prior[j,:,0]=320-1-prior[j,:,0]
    output.mkdir(parents=True)
    joblib.dump({'model':model,'feature_contract':'canonical_binary_mask_bbox_48x48_area_and_aspect_v2' if canonical else 'binary_mask_bbox_48x48_area_and_aspect_v1',
                 'landmark_order':ORDER,'source_commit':metadata['source_commit'],
                 'data_license':metadata['data_license'],'clinical_validation':False,
                 'input_domain':'orthographic cadaver femur silhouette; no DXA validation'},output/'candidate.joblib')
    report={'protocol':'15 training / 4 held-out subjects; all seven projections grouped by subject; fixed ExtraTrees128 leaf2 seed17; no held-out tuning',
            'test_subjects':sorted({rows[i]['subject'] for i in test}),
            'candidate':measures(prediction,truths[test],test,rows),
            'training_mean_layout_prior':measures(prior,truths[test],test,rows),
            'dataset_sha256':hashlib.sha256((dataset/'dataset.json').read_bytes()).hexdigest(),
            'image_sha256':hashes,'model_sha256':hashlib.sha256((output/'candidate.joblib').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'clinical_validation':False,'requirements_complete':False,'release_modified':False,
            'limitations':metadata['limitations']+['Uses a known femur silhouette; upstream DXA segmentation errors unassessed.',
                                                 'No pelvic superposition; one side per donor; held-out test only four cadavers.',
                                                 'Fixed proximal crop; robustness to clinical acquisition geometry not established.']}
    report['canonical_orientation'] = canonical
    report['evaluation_status'] = 'exploratory follow-up on previously inspected holdout; no new independent test' if canonical else 'initial fixed protocol'
    report['per_view'] = {}
    for angle in sorted({rows[i]['rotation_about_native_z_deg'] for i in test}):
        ids = np.array([j for j,i in enumerate(test) if rows[i]['rotation_about_native_z_deg'] == angle])
        report['per_view'][str(angle)] = measures(prediction[ids], truths[test[ids]], test[ids], rows)
    report['test_predictions'] = [{'subject': rows[i]['subject'], 'side': rows[i]['side'],
                                   'image': rows[i]['image'], 'prediction_points': prediction[j].tolist(),
                                   'reference_points': truths[i].tolist()} for j,i in enumerate(test)]
    (output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('candidate','training_mean_layout_prior')},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--canonical',action='store_true')
    args=parser.parse_args()
    run(args.dataset,args.output,args.canonical)
