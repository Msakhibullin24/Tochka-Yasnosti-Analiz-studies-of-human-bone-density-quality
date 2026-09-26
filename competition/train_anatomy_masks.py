"""Train automatic named masks from published annotations, no oracle input prompt."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from anatomy_training_data import records, load_record, split_records
from dxaqc.embedding import WEIGHTS_SHA256
from dxaqc.learned_anatomy import CLASSES, mask_head, spatial_features
from evaluate_organizer_dataset import sha256


def metrics(predicted, reference):
    known = reference != -100
    entries = []
    for label in np.unique(reference[known]):
        if label == 0:
            continue
        a = reference == label
        b = (predicted == label) & known
        intersection = int((a & b).sum())
        entries.append({'name': CLASSES[label], 'dice': 2*intersection/max(1, int(a.sum()+b.sum())),
                        'iou': intersection/max(1, int((a | b).sum()))})
    return entries


def run(args):
    if args.output.exists():
        raise ValueError('Choose a new training directory')
    torch.set_num_threads(4)
    torch.manual_seed(17)
    rows, excluded = records(args.aasce, args.dxa)
    test = split_records(rows)
    args.output.mkdir(parents=True)
    cache = np.lib.format.open_memmap(args.output/'features.npy', mode='w+', dtype=np.float16,
                                    shape=(len(rows), 896, 40, 40))
    targets = []
    for i, row in enumerate(rows):
        pixels, target = load_record(row)
        cache[i] = spatial_features(pixels)
        targets.append(cv2.resize(target.astype(np.float32), (40, 40), interpolation=cv2.INTER_NEAREST).astype(np.int64))
        row['split'] = 'test' if test[i] else 'train'
        if (i+1) % 25 == 0:
            print(f'Encoded {i+1}/{len(rows)}', flush=True)
    cache.flush()
    targets = np.asarray(targets)
    (args.output/'inventory.json').write_text(json.dumps({'rows': rows, 'excluded': excluded}, indent=2)+'\n')
    model = mask_head()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(17)
    train = np.flatnonzero(~test)
    dxa_train = np.array([i for i in train if rows[i]['source'] == 'ramathibodi'])
    # Fixed source-balanced schedule, chosen before reading held-out metrics.
    schedule = np.r_[train, np.tile(dxa_train, 16)]
    weights = torch.tensor([.2] + [1.] * 17)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        order = rng.permutation(schedule)
        for start in range(0, len(order), 8):
            ids = order[start:start+8]
            x = torch.from_numpy(np.array(cache[ids], dtype=np.float32))
            y = torch.from_numpy(targets[ids].copy())
            # Apply identical shifts to feature and annotation grids. This is
            # feature-space augmentation, not a physically simulated DXA image.
            shift = int(rng.integers(-3, 4))
            x = torch.roll(x, shift, dims=3)
            y = torch.roll(y, shift, dims=2)
            if shift > 0:
                x[:, :, :, :shift] = 0; y[:, :, :shift] = -100
            elif shift < 0:
                x[:, :, :, shift:] = 0; y[:, :, shift:] = -100
            x = x * float(rng.uniform(.8, 1.2))
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights, ignore_index=-100)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        print(f'Epoch {epoch+1}/{args.epochs}: {np.mean(losses):.4f}', flush=True)
    model.eval()
    cases = []
    # Layout baseline is derived only from known training pixels per source.
    priors = {}
    for source in sorted({row['source'] for row in rows}):
        source_train = [i for i in train if rows[i]['source'] == source]
        counts = np.array([(targets[source_train] == k).sum(axis=0) for k in range(len(CLASSES))])
        priors[source] = counts.argmax(axis=0)
    with torch.inference_mode():
        for i in np.flatnonzero(test):
            prediction = model(torch.from_numpy(np.array(cache[i], dtype=np.float32))[None])[0].argmax(0).numpy()
            cases.append({'source': rows[i]['source'], 'group': rows[i]['group'],
                          'image_sha256': rows[i]['image_sha256'],
                          'learned': metrics(prediction, targets[i]),
                          'layout_prior': metrics(priors[rows[i]['source']], targets[i])})
    path = args.output/'anatomy_masks.pt'
    torch.save({'schema_version': 1, 'classes': CLASSES, 'encoder_sha256': WEIGHTS_SHA256,
                'input_size': 320, 'grid': 40, 'state_dict': model.state_dict(),
                'clinical_validation': False}, path)
    summary = {}
    for source in sorted(priors):
        selected = [case for case in cases if case['source'] == source]
        summary[source] = {name+'_dice': float(np.mean([v['dice'] for c in selected for v in c[name]]))
                           for name in ('learned', 'layout_prior')}
        summary[source]['held_out_images'] = len(selected)
    report = {'protocol': 'Fixed source-stratified group holdout, seed17; fixed epochs and source balancing; no oracle prompts',
              'epochs': args.epochs, 'summary': summary, 'cases': cases,
              'training_images': int((~test).sum()), 'held_out_images': int(test.sum()),
              'excluded_annotations': excluded, 'model_sha256': sha256(path),
              'code_sha256': sha256(Path(__file__)),
              'adapter_sha256': sha256(Path(__file__).with_name('anatomy_training_data.py')),
              'clinical_validation': False, 'release_modified': False,
              'limitations': ['AASCE patient identity is unavailable: holdout is image-disjoint only.',
                              'DXA supervision is named vendor ROI, not independently annotated bone contours.',
                              'Metrics are measured on the 40x40 label grid; no physical-mm accuracy claimed.',
                              'Only two Hologic patients held out; GE transfer and hip landmarks not established.',
                              'Feature-space augmentation is not calibrated CT/DXA simulation.']}
    (args.output/'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aasce', 'dxa', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100:
        parser.error('epochs must be 1..100')
    run(args)
