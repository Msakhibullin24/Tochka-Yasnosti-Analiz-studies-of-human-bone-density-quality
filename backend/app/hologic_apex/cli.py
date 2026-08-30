from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from .archive import materialize_source
from .exporter import export_dataset
from .loader import load_dataset
from .parser import ApexFormatError


PROTOCOLS = (
    "spine_pa",
    "hip_left",
    "hip_right",
    "forearm_left",
    "forearm_right",
    "total_body",
    "unsupported",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osseo-apex",
        description="Audit and de-identify Hologic APEX P/R archives without modifying the source.",
    )
    parser.add_argument("input", type=Path, help="RAR archive or extracted directory")
    parser.add_argument("--output", type=Path, help="Create an anonymized PNG/NPY dataset at this new path")
    parser.add_argument(
        "--protocol",
        action="append",
        choices=PROTOCOLS,
        help="Export only this protocol; repeat to select multiple protocols",
    )
    parser.add_argument(
        "--key-env",
        default="OSSEO_PSEUDONYM_KEY",
        help="Environment variable containing the pseudonymization secret for export",
    )
    parser.add_argument(
        "--split-seed",
        default="osseo-dataset-v1",
        help="Stable non-secret seed for deterministic patient-grouped dataset splits",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    secret = os.getenv(args.key_env, "")
    if args.output and len(secret.encode("utf-8")) < 16:
        print(
            f"error: {args.key_env} must contain at least 16 UTF-8 bytes when --output is used",
            file=sys.stderr,
        )
        return 2
    key = secret.encode("utf-8") if secret else secrets.token_bytes(32)
    try:
        with materialize_source(args.input) as root:
            dataset = load_dataset(root)
            if args.output:
                summary = export_dataset(
                    dataset,
                    args.output,
                    key,
                    protocols=set(args.protocol) if args.protocol else None,
                    split_seed=args.split_seed,
                )
            else:
                summary = dataset.summary(key)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (ApexFormatError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
