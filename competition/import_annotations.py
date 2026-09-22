"""Convert completed COCO annotations + explicit review ledger into a checked YOLO dataset.

All source image hashes, study splits, category mappings and review attestations are
checked. Missing annotations count as negatives only with a reviewed ledger entry.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

from dxaqc.dicom_io import read_any
from train_detector import inspect_manifest, annotation


def convert(packet, coco_path, ledger_path, task, output):
    if output.exists():raise ValueError('Choose a new dataset directory')
    source=json.loads((packet/'review-manifest.json').read_text())
    records={r['image']:r for r in source['samples']}
    with ledger_path.open(newline='') as f: ledger=list(csv.DictReader(f))
    if len({r['image_id'] for r in ledger}) != len(ledger):raise ValueError('Duplicate review ledger entry')
    review={r['image_id']:r for r in ledger}
    coco=json.loads(coco_path.read_text());categories=coco['categories']
    if not categories or len({c['id'] for c in categories})!=len(categories):raise ValueError('Invalid categories')
    category_map={c['id']:i for i,c in enumerate(categories)}
    names=[c['name'] for c in categories]
    keypoints=categories[0].get('keypoints',[]) if task=='pose' else []
    if task=='pose' and (not keypoints or any(c.get('keypoints')!=keypoints for c in categories)):
        raise ValueError('Pose categories must share the same named keypoints')
    images=coco['images']
    if len({im['id'] for im in images})!=len(images) or len({im['file_name'] for im in images})!=len(images):
        raise ValueError('Duplicate image IDs or filenames')
    by_image={im['id']:[] for im in images}
    for ann in coco['annotations']:
        if ann['image_id'] not in by_image or ann['category_id'] not in category_map:
            raise ValueError('Unknown image/category reference')
        if ann.get('iscrowd',0):raise ValueError('Crowd/RLE annotations require explicit instance review')
        by_image[ann['image_id']].append(ann)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='.annotations-') as temp_name:
        temp=Path(temp_name);(temp/'images').mkdir();(temp/'labels').mkdir();samples=[]
        for im in images:
            record=records.get(im['file_name'])
            if record is None:raise ValueError('Image is not in the original review packet')
            attestation=review.get(record['image_id'],{})
            if attestation.get('reviewed','').lower()!='true' or not attestation.get('reviewer_id','').strip():
                raise ValueError('Every image requires reviewed=true and a reviewer identifier')
            path=(packet/record['image']).resolve()
            if not path.is_relative_to(packet.resolve()):raise ValueError('Image escapes packet')
            # PNG export is uint8; read_any may normalize it again. Check the actual PNG
            # pixels directly against the source decoder hash using the stored file hash below.
            import hashlib
            if not record.get('png_sha256') or hashlib.sha256(path.read_bytes()).hexdigest()!=record['png_sha256']:
                raise ValueError('Image file changed since expert packet creation')
            img=read_any(path)
            h,w=img.pixels.shape
            if (w,h)!=(im['width'],im['height']):raise ValueError('Annotation dimensions differ from source')
            lines=[]
            for ann in by_image[im['id']]:
                cls=category_map[ann['category_id']]
                if task=='segment':
                    polygons=ann.get('segmentation',[])
                    if not isinstance(polygons,list) or len(polygons)!=1:
                        raise ValueError('Exactly one polygon per instance is required; disconnected/RLE masks are not silently merged')
                    points=polygons[0]
                    row=[cls]+[value/(w if i%2==0 else h) for i,value in enumerate(points)]
                else:
                    x,y,bw,bh=ann['bbox'];row=[cls,(x+bw/2)/w,(y+bh/2)/h,bw/w,bh/h]
                    if task=='pose':
                        points=ann.get('keypoints',[])
                        if len(points)!=3*len(keypoints):raise ValueError('Wrong number of keypoints')
                        for i in range(0,len(points),3):
                            px,py,visibility=points[i:i+3]
                            row += [px/w,py/h,visibility] if visibility else [0,0,0]
                text=' '.join(str(v) for v in row)
                annotation(text,task,len(names),len(keypoints));lines.append(text)
            image_name=f"images/{record['image_id']}.png";label_name=f"labels/{record['image_id']}.txt"
            shutil.copy2(path,temp/image_name)
            (temp/label_name).write_text('\n'.join(lines)+('\n' if lines else ''))
            samples.append({'image':image_name,'label':label_name,'study':record['study'],
                            'split':record['split'],'reviewed':True})
        manifest={'task':task,'modality':'DXA','names':names,'samples':samples,'keypoint_names':keypoints}
        (temp/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        audit,_=inspect_manifest(temp/'manifest.json',temp)
        (temp/'data-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        temp.rename(output)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('packet','coco','ledger','output'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--task',choices=['detect','segment','pose'],required=True)
    a=p.parse_args();print(json.dumps(convert(a.packet,a.coco,a.ledger,a.task,a.output),indent=2))
