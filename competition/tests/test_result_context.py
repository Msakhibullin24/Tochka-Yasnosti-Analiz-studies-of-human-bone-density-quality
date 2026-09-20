import json

import pydicom

from dxaqc.explain import write_sr
from dxaqc.result_context import model_verdict, assessment_notes
from dxaqc.validate_results import validate
from dxaqc.report import write_csv


def untyped_row():
    return {'path_to_study':'a','path_to_file':'a.dcm','study_uid':'1.2','image_uid':'1.2.3',
            'anatomical_region':'Поясничный отдел позвоночника','quality_class':1,'quality_prob':.8,
            'violation_type':'','violation_codes':'','violation_type_status':'undetermined',
            'processing_status':'Success','time_of_processing':.1,'decision_version':'3',
            'projection_assessment':json.dumps({'value':'unknown','status':'undetermined'}),
            'source_roi_assessment':json.dumps({'status':'absent'}),
            'anatomy_assessment':json.dumps({'landmarks':[{'name':'Th12','points':[]}]}),
            'review_reasons':json.dumps(['violation_type_undetermined'])}


def test_unknown_type_is_explicit_in_table_and_report(tmp_path):
    row=untyped_row();path=tmp_path/'results.csv';write_csv([row],path)
    result=validate(path,manifest=['a.dcm'])
    assert result['valid'] and any('typification requirement is incomplete' in w for w in result['warnings'])
    assert 'тип не установлен' in model_verdict(row)
    notes=' '.join(assessment_notes(row))
    assert 'Th12' in notes and 'не подтверждена' in notes and 'отсутствует' in notes


def test_sr_carries_limits_and_new_content_gets_new_uid(tmp_path):
    row=untyped_row()
    def write(name,value):
        path=tmp_path/name
        write_sr(value,'1.2','1.2.3','1.2.840.10008.5.1.4.1.1.1',path,'1.2.4')
        return pydicom.dcmread(path)
    a=write('a.dcm',row);repeat=write('repeat.dcm',{**row,'time_of_processing':99})
    b=write('b.dcm',{**row,'quality_class':0})
    assert a.SOPInstanceUID==repeat.SOPInstanceUID
    assert a.SOPInstanceUID!=b.SOPInstanceUID
    text=' '.join(str(e.value) for e in a.iterall() if e.keyword=='TextValue')
    assert 'тип не установлен' in text and 'Th12' in text and 'не подтверждена' in text
    assert 'Качественное исследование' not in text
    assert all(len(str(e.value))<=16 for e in a.iterall() if e.keyword=='CodeValue')
