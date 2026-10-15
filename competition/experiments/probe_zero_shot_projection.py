"""Target-free projection probe using MedSigLIP text/image similarity.

External folder assignment (BMD frontal, VFA lateral) supplies exploratory
reference labels; this is not a verified AP/lateral benchmark. Prompts are
fixed before inspecting results. All image pixels and checkpoints stay local.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dxaqc.dicom_io import read_dxa
from gpu_research.common import digest, letterbox, checkpoint_identity

PROMPTS = [
    'A frontal anteroposterior DXA image of the lumbar spine and pelvis.',
    'A frontal anteroposterior DXA image of the proximal femur and hip.',
    'A lateral side view DXA image of the thoracic and lumbar spine.',
    'A lateral side view X-ray image of the hip and proximal femur.',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new probe directory')
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.set_num_threads(4)
    from transformers import AutoProcessor, AutoModel
    identity = checkpoint_identity(args.model, 'siglip')
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
    text = processor.tokenizer(PROMPTS, padding='max_length', max_length=64, return_tensors='pt')
    text = {k: v.cuda() for k, v in text.items()}
    results = []
    with torch.inference_mode():
        for subset, target in (('BMD', 0), ('VFA', 1)):
            for path in sorted((args.dataset / subset / 'Annotation').rglob('*.dcm')):
                image = read_dxa(path)
                rgb, _ = letterbox(image.pixels, 448)
                values = processor.image_processor(images=rgb, return_tensors='pt', do_resize=False)['pixel_values']
                logits = model(pixel_values=values.cuda().bfloat16(), **text).logits_per_image.float()[0]
                # Averaging fixed view-specific captions avoids choosing the
                # winning prompt retrospectively. Scores are not calibrated.
                frontal, lateral = logits[:2].mean(), logits[2:].mean()
                score = float(torch.sigmoid(lateral - frontal))
                results.append({'path': str(path.relative_to(args.dataset)), 'source_file_sha256': digest(path),
                    'decoded_pixel_sha256': image.pixel_sha256, 'folder_assigned_lateral': target,
                    'lateral_similarity_score': score, 'lateral_prediction': int(score >= .5),
                    'caption_logits': logits.cpu().tolist()})
    y = [r['folder_assigned_lateral'] for r in results]
    p = [r['lateral_prediction'] for r in results]
    scores = [r['lateral_similarity_score'] for r in results]
    report = {'protocol': __doc__, 'images': len(results), 'prompts': PROMPTS,
        'f1_against_folder_assignment': f1_score(y, p), 'roc_auc_against_folder_assignment': roc_auc_score(y, scores),
        'tn_fp_fn_tp': confusion_matrix(y, p, labels=[0, 1]).ravel().tolist(),
        'results': results, 'model_identity': identity, 'code_sha256': digest(Path(__file__)),
        'model_source': 'https://huggingface.co/google/medsiglip-448',
        'labels_expert_verified': False, 'qc_labels_used': False, 'clinical_validation': False,
        'limitations': ['VFA and BMD differ by acquisition field, shape and appearance; projection is confounded with protocol.',
                        'No lateral hip reference images are available in this probe.',
                        'This is not an estimate on the organiser cohort or a validation of all projection classes.']}
    args.output.mkdir(parents=True)
    (args.output/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k in ('images','f1_against_folder_assignment','roc_auc_against_folder_assignment','tn_fp_fn_tp')}), flush=True)


if __name__ == '__main__':
    main()
