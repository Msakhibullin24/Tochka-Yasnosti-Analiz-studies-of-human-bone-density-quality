"""Checksum-pinned EOS lateral pelvis source; never relabel as lateral DXA."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import shutil
import tempfile
import zipfile

from pinned_archive import fetch_archive,safe_members

URL='https://ndownloader.figshare.com/files/41815440'
SIZE=287048034
MD5='7e2c4d4ec2929d0a062a819667af2d8e'


def run(root):
    archive=fetch_archive(root/'Imaging-dataset.zip',URL,SIZE,MD5)
    output=root/'lateral-pelvis'
    if output.exists():raise ValueError('Source subset already exists')
    staging=Path(tempfile.mkdtemp(prefix='.pelvic-lateral-',dir=root))
    try:
        (staging/'images').mkdir();records=[];names=set()
        with zipfile.ZipFile(archive) as z:
            for item in safe_members(z,1000,1_000_000_000):
                name=PurePosixPath(item.filename).name
                if not name.lower().endswith(('.jpg','.jpeg')):continue
                if name in names:raise ValueError('Ambiguous image basename')
                names.add(name)
                # Paper specifies patient number as the first filename component.
                match=re.match(r'^(\d+)(?:[^\d]|\.(?:jpg|jpeg)$)',name,re.I)
                if not match:raise ValueError('Unrecognized patient filename convention')
                raw=z.read(item);(staging/'images'/name).write_bytes(raw)
                records.append({'image':f'images/{name}','patient_group':match.group(1),
                                'source_sha256':hashlib.sha256(raw).hexdigest(),'view':'lateral',
                                'reference_basis':'publisher EOS sagittal pelvic radiograph cohort'})
        if len(records)!=115 or len({r['patient_group'] for r in records})!=93:
            raise ValueError('Image/patient counts differ from published cohort')
        report={'doi':'10.6084/m9.figshare.23820879.v1','paper':'https://www.nature.com/articles/s41597-024-04003-7',
                'license':'CC-BY-4.0','archive_url':URL,'archive_bytes':SIZE,'archive_md5':MD5,
                'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                'images':len(records),'patients':93,'records':records,
                'scope':'EOS lateral pelvis, not lateral DXA; no trochanter or rotation-subtype labels',
                'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (staging/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
        staging.rename(output)
        print(json.dumps({k:report[k] for k in ('images','patients','archive_sha256')},indent=2),flush=True)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
