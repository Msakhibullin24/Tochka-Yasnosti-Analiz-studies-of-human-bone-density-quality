"""Patient-held-out adaptation of BUU centers to weak named DXA bone-area ROIs.

Five fixed folds, fixed 20 epochs. No test feedback, clinical ground truth,
numbering by assignment, or L5 target invention. Research model only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.nn import functional as F

from dxaqc.dicom_io import read_dxa
from evaluate_external_roi_masks import read_seg_nrrd
from evaluate_organizer_dataset import sha256
from predict_spatial_anatomy import load
from train_spatial_anatomy import spatial_embedding, heatmap_targets, decode


def labeled_loss(logits, targets, visible):
    if logits.shape != targets.shape or visible.shape != logits.shape[:2] or not visible.any():
        raise ValueError('Invalid visible-level supervision')
    per_level = -(targets.flatten(2)*F.log_softmax(logits.flatten(2), dim=2)).sum(2)
    return per_level[visible].mean()


def metric(prediction, reference, heights):
    errors = np.linalg.norm(prediction-reference, axis=-1)/heights
    return {'bodies': int(errors.size), 'within_half_roi_bbox_height': int((errors <= .5).sum()),
            'numbered_roi_center_hit_rate': float((errors <= .5).mean()),
            'median_error_roi_heights': float(np.median(errors))}


def run(root, base, output):
    if output.exists():
        raise ValueError('Choose a new experiment directory')
    torch.set_num_threads(4)
    rows, features, targets, references, heights, sizes = [], [], [], [], [], []
    for patient in sorted((root/'Annotation').iterdir()):
        if not patient.is_dir():
            continue
        source = patient/'images/spine_image.dcm'
        annotation = patient/'segmentations/spine_image.seg.nrrd'
        image = read_dxa(source)
        masks = read_seg_nrrd(annotation, image.pixels.shape)
        centers, body_heights = [], []
        for level in range(1, 5):
            yy, xx = np.where(masks[f'Lumbar_{level}_bone_area'])
            if not len(xx):
                raise ValueError('Missing named ROI')
            centers.append([xx.mean(), yy.mean()]);body_heights.append(yy.max()-yy.min()+1)
        h, w = image.pixels.shape
        normalized = np.array(centers)/np.array([w, h])
        # Dummy L5 heatmap has zero visibility and contributes no gradient.
        targets.append(heatmap_targets(np.vstack([normalized, [.5, .5]])))
        features.append([spatial_embedding(image.pixels), spatial_embedding(255-image.pixels)])
        references.append(centers);heights.append(body_heights);sizes.append([w, h])
        rows.append({'patient_id': patient.name, 'source_sha256': sha256(source),
                     'pixel_sha256': image.pixel_sha256, 'annotation_sha256': sha256(annotation)})
    if len(rows) < 5 or len({r['pixel_sha256'] for r in rows}) != len(rows):
        raise ValueError('Need independent patient rasters for five folds')
    features, targets = np.array(features), np.array(targets)
    references, heights, sizes = map(np.asarray, (references, heights, sizes))
    output.mkdir(parents=True)
    prediction = np.zeros_like(references, dtype=float);baseline = prediction.copy();prior = prediction.copy()
    fold_ids = np.zeros(len(rows), dtype=int)

    def fit(indices, seed):
        torch.manual_seed(seed)
        model = load(base)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        variants = np.array([(i, v) for i in indices for v in range(2)])
        generator = np.random.default_rng(seed)
        for epoch in range(20):
            model.train()
            for batch in np.array_split(generator.permutation(variants), max(1, len(variants)//4)):
                x = torch.from_numpy(features[batch[:, 0], batch[:, 1]].astype(np.float32))
                target = torch.from_numpy(targets[batch[:, 0]])
                visible = torch.ones((len(batch), 5), dtype=torch.bool);visible[:, 4] = False
                loss = labeled_loss(model(x), target, visible)
                optimizer.zero_grad();loss.backward();optimizer.step()
        return model.eval()

    def inference(model, indices):
        result = []
        with torch.inference_mode():
            for i in indices:
                logits = model(torch.from_numpy(features[i].astype(np.float32)))
                centers = decode(logits.numpy()).mean(axis=0)[:4]*sizes[i]
                result.append(centers)
        return np.array(result)

    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=17).split(rows)):
        model = fit(train, 17+fold)
        prediction[test] = inference(model, test)
        baseline[test] = inference(load(base), test)
        prior[test] = (references[train]/sizes[train, None]).mean(axis=0)*sizes[test, None]
        fold_ids[test] = fold
        print(f'Completed patient fold {fold+1}/5', flush=True)
    model = fit(np.arange(len(rows)), 22)
    bundle = torch.load(base, map_location='cpu', weights_only=True)
    bundle.update(state_dict=model.state_dict(), adaptation='Named bone-area ROI centers L1-L4 only',
                  l5_supervised_on_dxa=False, training_patients=len(rows), clinical_validation=False)
    torch.save(bundle, output/'spatial_dxa_head.pt')
    difference = ((np.linalg.norm(prediction-references, axis=-1)/heights <= .5).mean(axis=1)
                  -(np.linalg.norm(baseline-references, axis=-1)/heights <= .5).mean(axis=1))
    boot = np.random.default_rng(17).choice(difference, (1000, len(rows)), replace=True).mean(axis=1)
    report = {'protocol': 'Five patient folds; BUU initialization; only other eight DXA patients fitted per fold; original/inverted stay grouped; fixed 20 epochs, lr .0003; held-out targets used only for evaluation',
              'clinical_validation': False, 'release_modified': False,
              'patients': len(rows), 'adapted_oof': metric(prediction, references, heights),
              'frozen_buu_same_inference': metric(baseline, references, heights),
              'train_only_roi_layout': metric(prior, references, heights),
              'paired_patient_bootstrap_delta_hit_rate_ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
              'base_model_sha256': sha256(base), 'model_sha256': sha256(output/'spatial_dxa_head.pt'),
              'code_sha256': sha256(Path(__file__)),
              'cases': [{**r, 'fold': int(f), 'reference': a.tolist(), 'prediction': p.tolist(),
                         'baseline': b.tolist(), 'roi_heights': h.tolist()}
                        for r, f, a, p, b, h in zip(rows, fold_ids, references, prediction, baseline, heights)],
              'limitations': ['Ten patients, one Hologic source, weak ROI-center targets rather than vertebral anatomy.',
                              'No GE annotated transfer, L5 DXA, Th12, crest, disc or clinical QC ground truth.',
                              'All-patient final head is not the head evaluated in OOF; OOF score cannot attest its individual predictions.',
                              'No BUU retention check after adaptation; deployment not justified.']}
    (output/'evaluation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'limitations')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'base', 'output'):
        p.add_argument('--'+key, required=True, type=Path)
    a = p.parse_args()
    run(a.root, a.base, a.output)
