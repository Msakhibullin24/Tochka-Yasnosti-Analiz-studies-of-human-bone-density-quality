"""Smoke-test the training device without downloading weights or loading patient data."""
import argparse
import json
import platform
import time

import torch
import torchvision

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
args = parser.parse_args()
if args.device == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable: check NVIDIA driver and CUDA-enabled PyTorch installation')
torch.manual_seed(17)
torch.set_num_threads(2)
device = torch.device(args.device)
model = torchvision.models.resnet18(weights=None).to(device).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
x = torch.randn(2, 3, 224, 224, device=device)
y = torch.tensor([0, 1], device=device)
start = time.perf_counter()
use_bf16 = args.device == 'cuda' and torch.cuda.is_bf16_supported()
with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=use_bf16):
    loss = torch.nn.functional.cross_entropy(model(x), y)
loss.backward()
if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
    raise SystemExit('Non-finite loss or gradient')
optimizer.step()
if args.device == 'cuda':
    torch.cuda.synchronize()
print(json.dumps({
    'status': 'passed', 'purpose': 'random-weight forward/backward smoke test, not trained QC model',
    'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
    'device': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
    'cuda_runtime': torch.version.cuda,
    'capability': list(torch.cuda.get_device_capability()) if args.device == 'cuda' else None,
    'bf16': use_bf16, 'seconds': round(time.perf_counter() - start, 3),
    'peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
    'loss': float(loss.detach().cpu()),
}, indent=2))
