"""Package immutable spine/hip QC candidates for the existing shadow inference path."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from dxaqc.specialist_qc import SpecialistQC, RegionalSpecialists, digest


def pack(spine, hip, output):
    if output.exists(): raise ValueError('Choose a new suite directory')
    sources = {'spine':spine,'hip':hip}
    models = {k:SpecialistQC(v) for k,v in sources.items()}
    if any(m.metadata.get('region_scope') != k for k,m in models.items()):
        raise ValueError('Each candidate must match its target region')
    if len({m.metadata['labels_sha256'] for m in models.values()}) != 1:
        raise ValueError('Regional candidates must use the same source labels')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='.suite-') as tmp:
        temp = Path(tmp)
        spec = {'schema_version':1,'mode':'shadow','affects_decision':False,'members':{}}
        for region,path in sources.items():
            target=temp/region; target.mkdir()
            # Runtime needs only these two files; training/CAM artifacts stay in the source run.
            for name in ('model.json','model.ts'): shutil.copy2(path/name,target/name)
            spec['members'][region]={'path':region,'metadata_sha256':digest(target/'model.json')}
        (temp/'suite.json').write_text(json.dumps(spec,indent=2)+'\n')
        RegionalSpecialists(temp)
        temp.rename(output)
    return spec


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spine',type=Path,required=True);p.add_argument('--hip',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();pack(a.spine,a.hip,a.output)
