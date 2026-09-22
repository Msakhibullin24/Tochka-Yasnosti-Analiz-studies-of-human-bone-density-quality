import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip('matplotlib')

from report_specialists import measures, intervals, load_run
from train_detector import annotation, inspect_manifest


def test_missing_class_or_threshold_does_not_become_zero_metric():
    m = measures(np.array([0,0]), np.array([.1,.2]), None)
    assert m['roc_auc'] is None and m['f1'] is None
    assert m['confusion_tn_fp_fn_tp'] is None
    m = measures(np.array([0,0]), np.array([.1,.2]), .5)
    assert m['recall'] is None and m['precision'] is None
    assert m['specificity'] == 1 and m['accuracy'] == 1
    perfect = measures(np.array([0,1]), np.array([.1,.9]), .5)
    assert perfect['f1'] == perfect['roc_auc'] == 1
    assert perfect['confusion_tn_fp_fn_tp'] == [1,0,0,1]
    ci = intervals(np.array([0,0,1,1]), np.array([.1,.2,.8,.9]), .5, np.array(['a','a','b','b']), 100)
    assert ci['roc_auc']['low'] == ci['roc_auc']['high'] == 1
    assert ci['roc_auc']['defined_resamples'] < 100


@pytest.mark.parametrize('text,task', [('0 .5 .5 0 .2','detect'), ('0 .9 .5 .4 .4','detect'),
    ('1 .5 .5 .2 .2','detect'), ('0 nan .5 .2 .2','detect'),
    ('0 .1 .1 .2 .2 .3 .3','segment'), ('0 .1 .1 .2 .2','segment')])
def test_invalid_localization_labels_rejected(text,task):
    with pytest.raises(ValueError): annotation(text,task,1)


def fixture_manifest(root):
    rows = []
    for i,split in enumerate(('train','val','test')):
        cv2.imwrite(str(root/f'{i}.png'), np.random.default_rng(i).integers(0,256,(64,64),dtype=np.uint8))
        (root/f'{i}.txt').write_text('0 .5 .5 .2 .2\n')
        rows.append({'image':f'{i}.png','label':f'{i}.txt','study':str(i),'split':split,'reviewed':True})
    spec = {'task':'detect','modality':'DXA','names':['implant'],'samples':rows}
    p = root/'manifest.json'; p.write_text(json.dumps(spec))
    return p,spec


def test_detector_data_contract_rejects_leakage_missing_review_and_duplicates(tmp_path):
    p,spec = fixture_manifest(tmp_path)
    audit, prepared = inspect_manifest(p,tmp_path)
    assert audit['counts']['test']['objects'] == [1] and len(prepared) == 3
    spec['samples'][1]['study'] = '0'; p.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='Study leakage'): inspect_manifest(p,tmp_path)
    spec['samples'][1]['study'] = '1'; spec['samples'][1]['reviewed'] = False
    p.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='reviewed'): inspect_manifest(p,tmp_path)
    spec['samples'][1]['reviewed'] = True; spec['samples'][1]['image'] = '0.png'
    p.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='Duplicate'): inspect_manifest(p,tmp_path)


def test_empty_annotation_is_only_allowed_as_reviewed_negative(tmp_path):
    p,spec = fixture_manifest(tmp_path)
    (tmp_path/'1.txt').write_text('')
    with pytest.raises(ValueError, match='positive objects'): inspect_manifest(p,tmp_path)
    (tmp_path/'1.txt').unlink()
    with pytest.raises(ValueError, match='Missing file'): inspect_manifest(p,tmp_path)


def test_detector_training_evaluates_selected_checkpoint_on_separate_test(tmp_path,monkeypatch):
    import train_detector as training
    ultralytics = pytest.importorskip('ultralytics')
    p,_ = fixture_manifest(tmp_path)
    calls=[]
    class FakeYOLO:
        def __init__(self,path):
            self.path=path
            import torch
            self.model=torch.nn.Linear(1,1)
        def train(self,**kwargs):
            calls.append(('train',kwargs))
            best=tmp_path/'selected.pt';best.write_bytes(b'checkpoint')
            self.trainer=SimpleNamespace(best=best)
        def val(self,**kwargs):
            calls.append(('val',kwargs,self.path))
            return SimpleNamespace(results_dict={'metrics/mAP50(B)':.5},summary=lambda:[])
    catalog = json.loads((Path(training.__file__).resolve().parents[1]/'docs/competition/specialist_sources.json').read_text())
    expected = next(a['sha256'] for a in catalog['artifacts'] if a['id']=='yolo26x_weights')
    original_digest = training.digest
    monkeypatch.setattr(training,'digest',lambda p:expected if p.name=='yolo26x.pt' else original_digest(p))
    monkeypatch.setattr(ultralytics,'YOLO',FakeYOLO)
    args=SimpleNamespace(output=tmp_path/'run',epochs=1,batch=2,threads=1,size=64,
                         manifest=p,dataset=tmp_path,assets=tmp_path/'assets',device='cpu')
    report=training.run(args)
    assert calls[0][0]=='train' and calls[1][1]['split']=='val' and calls[2][1]['split']=='test'
    assert calls[2][2]==str(args.output/'best.pt')
    assert report['clinical_validation'] is False
    assert json.loads((args.output/'status.json').read_text())['status']=='completed'
    with pytest.raises(ValueError,match='overwrite'): training.run(args)


def test_report_checks_label_identity_and_prediction_groups(tmp_path):
    import hashlib
    import pandas as pd
    from dxaqc.specialist_qc import OUTPUTS, digest
    from report_specialists import build
    labels = pd.DataFrame({'study_key':['a','b','c'], 'quality_class':[0,1,0],
                           'region':['spine']*3, 'first_source_path':['a','b','c'],
                           'pixel_sha256':['a','b','c'],
                           **{n:[0,1,0] for n in OUTPUTS[1:]}})
    p=tmp_path/'labels.csv'; labels.to_csv(p,index=False)
    run=tmp_path/'model';run.mkdir()
    meta={'labels_sha256':digest(p),'thresholds':{n:.5 for n in OUTPUTS},
          'backbone':'test','loss':'bce','epochs':1,'split':{},'model_sha256':'test'}
    (run/'model.json').write_text(json.dumps(meta))
    rows=[{'label_row':i,'split':split,'study_hash':hashlib.sha256(g.encode()).hexdigest(),
           **{n+'_score':.1 for n in OUTPUTS}} for i,(g,split) in enumerate(zip('abc',['train','validation','test']))]
    (run/'predictions.json').write_text(json.dumps(rows))
    report=build([run],p,tmp_path/'report',10)
    assert (tmp_path/'report/index.html').is_file()
    assert report['runs'][0]['metrics']['quality']['test']['roc_auc'] is None
    rows[0]['study_hash']='wrong'; (run/'predictions.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='pairing'): load_run(run,p)
    p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='checksum'): load_run(run,p)


@pytest.mark.parametrize('task', ['detect', 'segment', 'pose'])
def test_real_yolo_training_contract_when_requested(tmp_path, task):
    """Synthetic fixture checks execution only, never DXA accuracy or medical labels."""
    import os
    if os.environ.get('DXAQC_TEST_YOLO_TRAIN') != '1':
        pytest.skip('Opt-in CPU integration against downloaded YOLO26x checkpoint')
    pytest.importorskip('ultralytics')
    import train_detector as training
    p,spec = fixture_manifest(tmp_path)
    spec['task'] = task
    if task == 'pose': spec['keypoint_names']=['a','b']
    p.write_text(json.dumps(spec))
    if task == 'pose':
        for i in range(3):
            (tmp_path/f'{i}.txt').write_text('0 .5 .5 .8 .8 .3 .3 2 .7 .7 2\n')
    if task == 'segment':
        for i in range(3):
            (tmp_path/f'{i}.txt').write_text('0 .3 .3 .7 .3 .7 .7 .3 .7\n')
    args=SimpleNamespace(output=tmp_path/'synthetic-contract-only',epochs=1,batch=1,threads=4,size=64,
                         manifest=p,dataset=tmp_path,assets=Path('data/specialists'),device='cpu')
    result=training.run(args)
    assert (args.output/'best.pt').is_file()
    assert set(result['metrics']) == {'val','test'}


def test_numpy_and_undefined_detector_metrics_are_json_safe():
    from train_detector import json_value
    result = json_value({'instances': np.int64(2), 'score': np.float32(.5), 'undefined': float('nan')})
    assert json.loads(json.dumps(result, allow_nan=False)) == {'instances':2, 'score':.5, 'undefined':None}
