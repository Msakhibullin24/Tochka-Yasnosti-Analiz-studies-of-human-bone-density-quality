import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_assets import verify


def make_assets(tmp_path):
    root=tmp_path/'assets';root.mkdir()
    code=tmp_path/'code';code.mkdir()
    (root/'weights.bin').write_bytes(b'reviewed model')
    (code/'inference.py').write_bytes(b'reviewed inference')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'includes_medical_images':False,'includes_qc_targets':False,
              'files':{'weights.bin':digest(root/'weights.bin')}}
    (root/'assets_manifest.json').write_text(json.dumps(manifest))
    policy=tmp_path/'policy.json'
    policy.write_text(json.dumps({'assets_manifest_sha256':digest(root/'assets_manifest.json'),
          'metric_scope':'synthetic integrity fixture only',
          'runtime_code_sha256':{'inference.py':digest(code/'inference.py')}}))
    return root,policy,code


def test_reviewed_models_and_code_are_accepted(tmp_path):
    root,policy,code=make_assets(tmp_path)
    result=verify(root,policy,code)
    assert result['approved'] and result['verified_files']==result['verified_runtime_modules']==1


@pytest.mark.parametrize('change', ['weights','extra','manifest','code'])
def test_changed_artifacts_never_pass_by_reusing_old_manifest(tmp_path,change):
    root,policy,code=make_assets(tmp_path)
    if change=='weights':(root/'weights.bin').write_bytes(b'different model')
    elif change=='extra':(root/'training.dcm').write_bytes(b'unexpected data')
    elif change=='manifest':(root/'assets_manifest.json').write_text('{}')
    else:(code/'inference.py').write_bytes(b'different inference')
    with pytest.raises(ValueError):verify(root,policy,code)
