"""Check every locked GPU package in a reused inference virtual environment."""
from __future__ import annotations
import argparse
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def verify(lock):
    checked = 0
    for line in Path(lock).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==', 1)
        try:
            actual = version(name)
        except PackageNotFoundError:
            raise ValueError('Locked package missing: ' + name) from None
        if actual != expected:
            raise ValueError(f'Locked package differs: {name}: {actual} != {expected}')
        checked += 1
    if not checked:
        raise ValueError('Empty environment lock')
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    args = parser.parse_args()
    print(f'Checked {verify(args.lock)} locked packages', flush=True)
