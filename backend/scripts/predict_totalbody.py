"""Explicit total-body research inference using the pinned 105-landmark checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fetch_dxa_checkpoint import EXPECTED_SHA256
from app.settings import settings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Choose a new output file')
    if hashlib.sha256(settings.checkpoint_path.read_bytes()).hexdigest()!=EXPECTED_SHA256:
        raise ValueError('Whole-body checkpoint integrity mismatch')
    import numpy as np
    import torch
    from PIL import Image
    from app.inference import DxaPointPlacementModel
    torch.set_num_threads(4)
    image=np.array(Image.open(a.image).convert('L'))
    points=DxaPointPlacementModel().predict(image)
    result={'model':'hawaii-ai/dxa-pointplacement','protocol':'whole-body-air-ratio',
            'clinical_validation':False,'affects_decision':False,'landmarks':[p.as_dict() for p in points],
            'limitation':'Research landmarks; source image protocol must be established separately.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
