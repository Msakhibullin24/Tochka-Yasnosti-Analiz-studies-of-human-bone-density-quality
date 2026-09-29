"""Acquire a checksum-pinned public hip/projection subset of FracAtlas.

CC BY 4.0 data, DOI 10.6084/m9.figshare.22363012. Ordinary radiographs,
not DXA. Author view tags can include multiple views; do not silently relabel.
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
import zipfile
from collections import Counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.pinned_archive import fetch_archive, safe_members

URL='https://ndownloader.figshare.com/files/65518038'
SIZE=338436460
MD5='fe9da2c7c285915ebee69dfdab8fd396'


def acquire(root):
    root.mkdir(parents=True,exist_ok=True)
    archive=fetch_archive(root/'FracAtlas.zip',URL,SIZE,MD5)
    output=root/'hip-subset'
    if output.exists():raise ValueError('Hip subset already exists; choose a new root')
    with zipfile.ZipFile(archive) as z:
        entries=safe_members(z)
        csvs=[i for i in entries if PurePosixPath(i.filename).name=='dataset.csv']
        if len(csvs)!=1 or csvs[0].file_size>2_000_000:raise ValueError('Ambiguous or oversized source CSV')
        raw=z.read(csvs[0]);rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        required={'image_id','hip','frontal','lateral','oblique','mixed','multiscan'}
        if not rows or not required<=rows[0].keys():raise ValueError('Missing projection/source labels')
        rows=[r for r in rows if r['hip']=='1']
        expected={r['image_id'] for r in rows}
        if not rows or len(expected)!=len(rows):raise ValueError('Empty or duplicate hip CSV records')
        if any(PurePosixPath(name).name!=name or '\\' in name for name in expected):raise ValueError('Unsafe CSV image identifier')
        files={} 
        for item in entries:
            filename=PurePosixPath(item.filename).name
            if filename in expected:
                if filename in files:raise ValueError('Ambiguous hip image filename')
                files[filename]=item
        if set(files)!=expected:raise ValueError('Missing source hip images')
        if sum(i.file_size for i in files.values())>200_000_000:raise ValueError('Hip subset exceeds extraction budget')
        staging=Path(tempfile.mkdtemp(prefix='.fracatlas-hips-',dir=root))
        try:
            (staging/'images').mkdir()
            (staging/'source-dataset.csv').write_bytes(raw)
            for row in rows:
                filename=row['image_id']
                content=z.read(files[filename]);(staging/'images'/filename).write_bytes(content)
                row['source_sha256']=hashlib.sha256(content).hexdigest()
            with (staging/'hips.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            staging.rename(output)
        except BaseException:
            shutil.rmtree(staging,ignore_errors=True);raise
    counts=Counter(tuple(r[k] for k in ('frontal','lateral','oblique','mixed','multiscan')) for r in rows)
    report={'doi':'10.6084/m9.figshare.22363012','license':'CC-BY-4.0','archive_url':URL,
            'archive_bytes':SIZE,'archive_md5':MD5,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'hip_images':len(rows),'view_combinations':{'/'.join(k):v for k,v in counts.items()},
            'columns_order':['frontal','lateral','oblique','mixed','multiscan'],
            'scope':'Author-labelled ordinary hip radiographs; no DXA QC/rotation-subtype reference',
            'patient_identity_available':False}
    (root/'acquisition.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    acquire(p.parse_args().output)
