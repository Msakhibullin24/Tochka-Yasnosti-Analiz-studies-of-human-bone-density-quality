"""Descriptive regression guard on the same previously inspected holdout."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def compare_original_cases(baseline, candidate):
    def indexed(rows):
        result = {}
        for row in rows:
            if row['polarity'] != 'original':
                continue
            key = (row['source'], row['view'], row['group'])
            score = float(row['dice'])
            if key in result or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Duplicate holdout identity or invalid Dice')
            result[key] = score
        if not result:
            raise ValueError('No ordinary-field holdout cases')
        return result
    before, after = indexed(baseline), indexed(candidate)
    if before.keys() != after.keys():
        raise ValueError('Candidate and baseline holdout identities differ')
    cohorts = defaultdict(list)
    for key in before:
        cohorts[key[:2]].append(key)
    comparison = {}
    for (source, view), keys in sorted(cohorts.items()):
        first = float(np.mean([before[k] for k in keys]))
        second = float(np.mean([after[k] for k in keys]))
        comparison[f'{source}/{view}'] = {'images': len(keys), 'baseline_mean_dice': first,
                                         'candidate_mean_dice': second, 'delta': second-first}
    regressions = [key for key, item in comparison.items() if item['delta'] < 0]
    return {'comparison': comparison, 'ordinary_field_regressions': regressions,
            'ordinary_fields_non_decreasing': not regressions,
            'scope': 'descriptive paired source-holdout guard; no statistical noninferiority or clinical validation',
            'clinical_validation': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--baseline-model', default='lumbar_refinement')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output report')
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    result = compare_original_cases([r for r in baseline['annotated'] if r['model']==args.baseline_model], candidate['cases'])
    result.update(baseline_report_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  candidate_report_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
