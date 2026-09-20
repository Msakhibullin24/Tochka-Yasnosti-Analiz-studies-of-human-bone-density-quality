"""Read-only decision/geometry consistency audit of an extended batch CSV."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit(path):
    with open(path,encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    counts=Counter();examples=defaultdict(list)
    for row in rows:
        if row['processing_status']!='Success':continue
        def parsed(key,default):return json.loads(row[key]) if row.get(key) else default
        states=parsed('criterion_states',{})
        features=parsed('measurements',{})
        violations=set(filter(None,row.get('violation_codes','').split(';')))
        angle=states.get('spine_axis',{}).get('angle_deg',features.get('spine_abs_angle_deg'))
        flags=[]
        if angle is not None:
            if angle>5 and 'spine_axis' not in violations:flags.append('axis_over_5_not_reported')
            if angle<=5 and 'spine_axis' in violations:flags.append('axis_under_5_reported')
        if row['quality_class']=='0' and any(s['status']=='fail' for s in states.values()):flags.append('failed_criterion_suppressed')
        if row.get('decision_reason')=='highest_scoring_criterion':flags.append('forced_type')
        if row['quality_class']=='1' and not violations:flags.append('type_undetermined')
        if row.get('decision_reason')=='implant_rule':flags.append('forced_implant_rule')
        for item in parsed('anatomy_assessment',{}).get('landmarks',[]):
            if item['name']=='greater_trochanter' and item.get('points') and item['points'][0][1]<=1:
                flags.append('trochanter_at_top_edge')
        for flag in flags:
            counts[flag]+=1
            examples[flag].append({'path':row.get('path_to_file'),'image_uid':row.get('image_uid'),'angle':angle})
    defects=['axis_over_5_not_reported','axis_under_5_reported','failed_criterion_suppressed','forced_type','forced_implant_rule','trochanter_at_top_edge']
    return {'rows':len(rows),'success':sum(r['processing_status']=='Success' for r in rows),
            'defects':{key:counts[key] for key in defects},'type_undetermined':counts['type_undetermined'],
            'examples':dict(examples),'results_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope':'internal consistency, including duplicates; no expert accuracy claim'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.results)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},ensure_ascii=False,indent=2))
    return int(any(result['defects'].values()))


if __name__=='__main__':raise SystemExit(main())
