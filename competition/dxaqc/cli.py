"""Command line batch processing.

    python -m dxaqc.cli --input /data/in (folder or .zip) --output /data/out [--no-explanations] [--strict-columns]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import Options, OutputConflictError, run_batch, write_failure_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DXA quality control: batch processing of DICOM studies")
    ap.add_argument("--input", required=True, type=Path, help="folder with studies or a .zip archive")
    ap.add_argument("--output", required=True, type=Path, help="folder for results.csv / results.xlsx / zip")
    ap.add_argument("--no-explanations", action="store_true", help="skip overlay / DICOM SC / DICOM SR series")
    ap.set_defaults(strict_columns=True)
    ap.add_argument("--extended-columns", dest="strict_columns", action="store_false", help="extended results.csv for diagnostics")
    ap.add_argument("--strict-columns", action="store_true", help="only the 8 contract columns of the task statement + quality_prob")
    ap.add_argument('--acceptance', choices=('report', 'submission', 'complete'), default='report',
                    help='report: processing only; submission: require valid table and no failures; '
                         'complete: also require all mandatory checks; rejection exits 3 and preserves reports')
    args = ap.parse_args(argv)
    try:
        summary = run_batch(args.input, args.output, Options(explanations=not args.no_explanations,
                                                             strict_columns=args.strict_columns),
                            progress=lambda i, n: print(f"\r{i}/{n}", end="", file=sys.stderr, flush=True))
    except OutputConflictError as exc:
        print(f"OUTPUT_CONFLICT: {exc}; previous results preserved", file=sys.stderr)
        return 2
    except Exception as exc:  # input-level problem (missing path, broken archive): report, do not crash
        message = f"{type(exc).__name__}: {exc}"
        print(f"\nFAILED: {message}", file=sys.stderr)
        try:  # the organiser must always find a well-formed table, even if it only explains the failure
            write_failure_report(args.input, args.output, exc, Options(strict_columns=args.strict_columns))
        except OSError as io_exc:
            print(f"cannot write to the output folder: {io_exc}", file=sys.stderr)
        return 2
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    if args.acceptance != 'report' and (not summary['submission_valid'] or summary['failure']):
        print('ACCEPTANCE_FAILED: invalid submission or processing failures; see reports', file=sys.stderr)
        return 3
    if args.acceptance == 'complete' and not summary['requirements_complete']:
        print('ACCEPTANCE_FAILED: mandatory checks incomplete; see requirements.json', file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
