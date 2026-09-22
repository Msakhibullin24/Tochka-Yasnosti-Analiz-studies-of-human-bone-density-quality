"""Read-only, offline specialist inventory. Integrity and clinical readiness are separate."""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
from .specialist_qc import digest


@lru_cache(maxsize=128)
def checked_hash(path, size, modified_ns):
    return digest(Path(path))


def inventory(assets: Path | None = None, catalog_path: Path | None = None):
    root=Path(__file__).resolve().parents[2]
    if catalog_path is None:
        choices=[root/'docs/competition/specialist_sources.json',Path(__file__).resolve().parents[1]/'specialist_sources.json']
        catalog_path=next((p for p in choices if p.is_file()),choices[0])
    assets=Path(assets or os.environ.get('DXAQC_SPECIALIST_ASSETS',root/'data/specialists')).resolve()
    if not catalog_path.is_file():
        return {'configured':False,'clinical_models_ready':False,'capabilities':[]}
    catalog=json.loads(catalog_path.read_text());states={}
    for spec in catalog['artifacts']:
        path=(assets/spec['path']).resolve()
        status='missing'
        if path.is_relative_to(assets) and path.is_file():
            stat=path.stat()
            status=('verified' if stat.st_size==spec['bytes'] and checked_hash(str(path),stat.st_size,stat.st_mtime_ns)==spec['sha256']
                    else 'corrupt')
        states[spec['id']]=status
    return {'configured':True,'clinical_models_ready':False,'capabilities':[
        {'id':c['id'],'protocol':c['protocol'],'task':c['task'],'readiness':c['readiness'],
         'limitation':c['limitation'],'artifacts':{a:states[a] for a in c['artifacts']},
         'artifacts_verified':all(states[a]=='verified' for a in c['artifacts'])}
        for c in catalog['capabilities']]}
