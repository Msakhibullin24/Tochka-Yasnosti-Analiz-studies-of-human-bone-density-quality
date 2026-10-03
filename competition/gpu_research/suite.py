"""Run explicitly selected GPU experiments sequentially; preserve failure logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_research.common import CATALOG, HERE, dump


def run(args):
    if args.output.exists():
        raise ValueError('Choose a new suite directory')
    args.output.mkdir(parents=True)
    commands = []
    def execute(command, name):
        commands.append({'step': name, 'argv': command, 'status': 'running'})
        dump(args.output / 'commands.json', commands)
        with (args.output / f'{name}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=HERE.parent, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                commands[-1]['status'] = 'interrupted'
                dump(args.output / 'commands.json', commands)
                raise
        commands[-1].update(status='passed' if code == 0 else 'failed', exit_code=code)
        dump(args.output / 'commands.json', commands)
        if code:
            raise RuntimeError(f'{name} failed ({code}); inspect {name}.log; no candidate promoted')
    python = sys.executable
    cli = str(Path(__file__).with_name('run.py'))
    execute([python, cli, 'doctor', '--output', str(args.output / 'hardware.json')], 'doctor')
    results = {}
    for name in args.models:
        model_dir = args.models_root / name
        if args.download:
            execute([python, cli, 'download', '--model', name, '--root', str(args.models_root)], f'{name}-download')
        if not (model_dir / 'config.json').exists():
            raise FileNotFoundError(f'Missing {model_dir}; first download this model or use --download')
        base = [python, cli, 'extract', '--model', name, '--model-dir', str(model_dir),
                '--dataset', str(args.dataset), '--labels', str(args.labels), '--precision', args.precision]
        execute([*base, '--output', str(args.output / f'{name}-smoke'), '--smoke'], f'{name}-smoke')
        if args.stage == 'frozen':
            feature_dir = args.output / f'{name}-features'
            execute([*base, '--output', str(feature_dir)], f'{name}-features')
            evaluation = args.output / f'{name}-nested'
            execute([python, str(HERE / 'experiments/nested_criterion_upgrade.py'),
                     '--dataset', str(args.dataset), '--labels', str(args.labels),
                     '--baseline', str(args.baseline), '--embeddings', str(feature_dir / 'features.npy'),
                     '--output', str(evaluation)], f'{name}-nested')
        else:
            evaluation = args.output / f'{name}-finetune'
            execute([python, str(Path(__file__).with_name('finetune.py')),
                     '--dataset', str(args.dataset), '--labels', str(args.labels),
                     '--baseline', str(args.baseline), '--model', name, '--model-dir', str(model_dir),
                     '--output', str(evaluation), '--precision', args.precision,
                     '--batch-size', str(args.batch_size), '--epochs', str(args.epochs),
                     '--seed', str(args.seed),
                     '--last-blocks', str(args.last_blocks)], f'{name}-finetune')
        results[name] = json.loads((evaluation / 'evaluation.json').read_text())
        dump(args.output / 'summary.json', {'stage': args.stage, 'results': results,
                                            'automatic_promotions': 0, 'clinical_validation': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('frozen', 'finetune'), default='frozen')
    p.add_argument('--models', nargs='+', choices=CATALOG, default=['medsiglip', 'dinov3-b'])
    p.add_argument('--models-root', type=Path, default=HERE.parent / 'data/gpu-models')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--labels', type=Path, default=HERE / 'labels/image_labels.csv')
    p.add_argument('--baseline', type=Path, default=HERE / 'reports/quality_review_axis_guard_posthoc_2026_09_29.csv')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--download', action='store_true', help='Download selected weights before offline inference')
    p.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--last-blocks', type=int, default=2)
    p.add_argument('--seed', type=int, default=17, help='Fine-tuning seed; frozen nested protocol keeps its fixed seed')
    args = p.parse_args()
    for name in ('models_root', 'dataset', 'labels', 'baseline', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if len(set(args.models)) != len(args.models):
        p.error('Each model must be selected once')
    run(args)


if __name__ == '__main__':
    main()
