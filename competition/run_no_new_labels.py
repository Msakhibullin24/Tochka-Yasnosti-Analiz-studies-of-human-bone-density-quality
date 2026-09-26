"""Run a checksum-pinned offline profile with anatomy, projection and typed review."""
import argparse
import json
import warnings
from pathlib import Path

from dxaqc.pipeline import MODEL_PATH, Options, run_batch


def configure(profile_path):
    from dxaqc.workflow_profile import configure_profile
    return configure_profile(profile_path, MODEL_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output', 'profile'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--no-explanations', action='store_true')
    args = parser.parse_args()
    # Original organiser UIDs are invalid; source identity is preserved. Avoid
    # printing thousands of identical format warnings during the batch run.
    warnings.filterwarnings('ignore', message='Invalid value for VR UI:.*')
    warnings.filterwarnings('ignore', message='TypedStorage is deprecated.*')
    profile = configure(args.profile)
    summary = run_batch(args.input, args.output, Options(explanations=not args.no_explanations,
                        strict_columns=True, keep_explanation_dir=True),
                        progress=lambda i,n: print(f'{i}/{n}', flush=True) if i%25 == 0 else None)
    summary['profile_id'] = profile['profile_id']
    (args.output/'workflow_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
