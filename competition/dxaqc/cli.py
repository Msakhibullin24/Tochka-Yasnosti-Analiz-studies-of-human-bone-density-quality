"""Command line batch processing.

    python -m dxaqc.cli --input /data/in (folder or .zip) --output /data/out [--no-explanations] [--strict-columns]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import Options, run_batch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DXA quality control: batch processing of DICOM studies")
    ap.add_argument("--input", required=True, type=Path, help="folder with studies or a .zip archive")
    ap.add_argument("--output", required=True, type=Path, help="folder for results.csv / results.xlsx / zip")
    ap.add_argument("--no-explanations", action="store_true", help="skip overlay / DICOM SC / DICOM SR series")
    ap.add_argument("--strict-columns", action="store_true", help="only the 8 contract columns of the task statement + quality_prob")
    args = ap.parse_args(argv)
    try:
        summary = run_batch(args.input, args.output, Options(explanations=not args.no_explanations,
                                                             strict_columns=args.strict_columns),
                            progress=lambda i, n: print(f"\r{i}/{n}", end="", file=sys.stderr, flush=True))
    except Exception as exc:  # input-level problem (missing path, broken archive): report, do not crash
        message = f"{type(exc).__name__}: {exc}"
        print(f"\nFAILED: {message}", file=sys.stderr)
        try:  # the organiser must always find a well-formed table, even if it only explains the failure
            from .report import write_csv
            args.output.mkdir(parents=True, exist_ok=True)
            write_csv([{"path_to_study": str(args.input), "quality_class": 1, "violation_type": "", "violation_codes": "processing_failure",
                        "processing_status": "Failure", "time_of_processing": 0.0, "error_code": "BATCH_INPUT_ERROR",
                        "error_message": message[:300]}], args.output / "results.csv", args.strict_columns)
            (args.output / "summary.json").write_text(json.dumps({"error": message}, ensure_ascii=False), encoding="utf-8")
        except OSError as io_exc:
            print(f"cannot write to the output folder: {io_exc}", file=sys.stderr)
        return 2
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
