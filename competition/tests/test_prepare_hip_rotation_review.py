import pytest

from dxaqc.model import VIOLATION_LABEL
from prepare_hip_rotation_review import select


def test_hip_review_selects_all_errors_and_balances_control_sides():
    definitions = [
        ('fn', 'hip_left', '1', False),
        ('fp', 'hip_right', '0', True),
        ('tp-left-a', 'hip_left', '1', True),
        ('tp-left-b', 'hip_left', '1', True),
        ('tp-right-a', 'hip_right', '1', True),
        ('tp-right-b', 'hip_right', '1', True),
        ('tn-left-a', 'hip_left', '0', False),
        ('tn-left-b', 'hip_left', '0', False),
        ('tn-right-a', 'hip_right', '0', False),
        ('tn-right-b', 'hip_right', '0', False),
    ]
    labels, rows = {}, []
    for name, region, target, predicted in definitions:
        path = name + '.dcm'
        labels[path] = {'study_key': name, 'region': region, 'quality_class': target,
                        'hip_position_rotation': target}
        rows.append({'source_path': path, 'study': name, 'true_region': region,
                     'quality_true': target, 'repeat': '0', 'fold': '0',
                     'violation_type': VIOLATION_LABEL['hip_position_rotation'] if predicted else ''})
    selected = select(rows, labels, controls_per_class=2)
    kinds = {item['source_path']: item['kind'] for item in selected}
    assert kinds['fn.dcm'] == 'false_negative'
    assert kinds['fp.dcm'] == 'false_positive'
    assert len(selected) == 6
    for kind in ('true_positive', 'true_negative'):
        assert {item['region'] for item in selected if item['kind'] == kind} == {'hip_left', 'hip_right'}
    rows[1]['study'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        select(rows, labels, controls_per_class=2)
