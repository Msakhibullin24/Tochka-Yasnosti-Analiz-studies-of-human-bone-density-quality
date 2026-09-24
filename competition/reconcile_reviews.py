"""Compare independent QC reviews without changing the organizer's labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from dxaqc.model import CRITERIA,group_of


def reconcile(packet,a_path,b_path,output):
    if output.exists():raise ValueError('Choose a new review comparison file')
    manifest_path = packet/'review-manifest.json'
    if not manifest_path.is_file():
        manifest_path = packet/'manifest.json'
    source=json.loads(manifest_path.read_text())
    samples=source.get('samples', source.get('cases'))
    if not isinstance(samples,list) or not samples:
        raise ValueError('Review packet has no images')
    ids={r['image_id'] for r in samples}
    if len(ids)!=len(samples):
        raise ValueError('Duplicate review image ID')
    def load(path):
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        if len(rows)!=len(ids) or {r['image_id'] for r in rows}!=ids:
            raise ValueError('Each review must cover exactly the packet images')
        return {r['image_id']:r for r in rows}
    a,b=load(a_path),load(b_path)
    results=[]
    for sample in samples:
        ident=sample['image_id'];left,right=a[ident],b[ident]
        item={'image_id':ident,'status':'pending','agreed_labels':{},'disagreements':[]}
        if left['reviewed'].lower()=='true' and right['reviewed'].lower()=='true':
            if not left['reviewer_id'].strip() or not right['reviewer_id'].strip() or left['reviewer_id']==right['reviewer_id']:
                raise ValueError('Independent reviews require two distinct named reviewers')
            applicable=['quality',*CRITERIA[group_of(sample['region'])]]
            for name in applicable:
                values=[r[name].strip() for r in (left,right)]
                if any(v not in ('','0','1') for v in values):raise ValueError('Labels must be 0, 1 or empty')
                if values[0] and values[0]==values[1]:item['agreed_labels'][name]=int(values[0])
                else:item['disagreements'].append(name)
            item['status']='agreed' if not item['disagreements'] else 'requires_adjudication'
            if item['status']=='agreed':
                quality=item['agreed_labels']['quality'];criteria=[item['agreed_labels'][n] for n in applicable[1:]]
                if quality!=int(any(criteria)):
                    item['status']='requires_adjudication';item['disagreements'].append('quality_vs_taxonomy')
        results.append(item)
    report={'source_labels_sha256':source.get('labels_sha256'),
            'review_packet_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'original_labels_changed':False,
            'counts':{s:sum(r['status']==s for r in results) for s in ('pending','agreed','requires_adjudication')},
            'images':results}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','review-a','review-b','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(reconcile(a.packet,a.review_a,a.review_b,a.output)['counts'])
