import csv
import json
from pathlib import Path

import pytest

from prepare_priority_anatomy_review import digest, prepare, select_cases


def _fixture(tmp_path: Path):
    package = tmp_path/'source'
    (package/'images').mkdir(parents=True)
    cases, rows = [], []
    for index in range(4):
        group = f'study-{index}'
        image = f'images/{index:04d}.png'
        (package/image).write_bytes(f'image-{index}'.encode())
        cases.append({'image_uid': f'uid-{index}', 'path_to_file': f'Исследования/{group}/image.dcm',
                      'study_group': f'dicom-hash-{index}', 'image': image, 'status': 'unreviewed',
                      'landmarks': {'Th12': {'visible': None, 'point': None}}})
        rows.append({'source_path': f'{group}/image.dcm', 'study': group,
                     'quality_true': '1', 'quality_pred': '0', 'quality_score': 'secret model score'})
    reference = {'version': 1, 'review_package_version': 4, 'results_sha256': 'results-hash',
                 'coordinate_system': 'original_pixel_centres', 'cases': cases}
    (package/'template.json').write_text(json.dumps(reference))
    (package/'manifest.json').write_text(json.dumps({'files_sha256': {
        str(path.relative_to(package)): digest(path) for path in package.rglob('*') if path.is_file()
    }}))
    oof = tmp_path/'oof.csv'
    with oof.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    return package, oof, reference, rows


def test_audit_selection_ignores_outcomes_and_keeps_studies_disjoint(tmp_path):
    _, _, reference, rows = _fixture(tmp_path)
    audit, development, _ = select_cases(reference, rows, audit_studies=1, seed=7)
    changed = [dict(row, quality_true='0', quality_pred='1') for row in rows]
    repeated, _, _ = select_cases(reference, changed, audit_studies=1, seed=7)
    assert [case['image_uid'] for case in audit] == [case['image_uid'] for case in repeated]
    assert {case['study_group'] for case in audit}.isdisjoint(
        {case['study_group'] for case in development})
    assert len(audit) == 1 and len(development) == 3


def test_published_packages_are_blind_and_bound_to_source_manifest(tmp_path):
    package, oof, _, _ = _fixture(tmp_path)
    output = tmp_path/'queues'
    assert prepare(package, oof, output, audit_studies=1, seed=7) == {
        'audit_cases': 1, 'development_cases': 3,
        'development_reasons': {'false_negative': 3}}
    audit = json.loads((output/'audit/template.json').read_text())
    development = json.loads((output/'development/template.json').read_text())
    assert {case['image_uid'] for case in audit['cases']}.isdisjoint(
        {case['image_uid'] for case in development['cases']})
    for name, selected in [('audit', audit), ('development', development)]:
        page = (output/name/'index.html').read_text()
        assert 'secret model score' not in page and 'quality_true' not in page
        assert len(selected['cases']) == json.loads((output/name/'manifest.json').read_text())['cases']
        assert all((output/name/case['image']).is_file() for case in selected['cases'])
    (package/development['cases'][0]['image']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs from source package manifest'):
        prepare(package, oof, tmp_path/'changed', audit_studies=1, seed=7)
    assert not (tmp_path/'changed').exists()
