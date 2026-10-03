import pytest

from audit_delivery import compare_rows, read_rows, source_manifest


def test_independent_manifest_is_bound_to_source_bytes_and_rejects_escape(tmp_path):
    root = tmp_path / 'input'
    root.mkdir()
    source = root / 'image.dcm'
    source.write_bytes(b'original')
    (root / 'notes.xlsx').write_bytes(b'not an image')
    first = source_manifest(root)
    assert len(first) == 1 and first[0]['path'] == 'image.dcm'
    source.write_bytes(b'changed')
    assert source_manifest(root)[0]['sha256'] != first[0]['sha256']
    outside = tmp_path / 'external.dcm'
    outside.write_bytes(b'external')
    (root / 'escape.dcm').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        source_manifest(root)


def test_batch_comparison_ignores_timing_but_checks_probability_and_axis():
    before = {'a.dcm': {'quality_class': '0', 'quality_prob': '.1', 'measurements': '{"angle": 0}',
                         'time_of_processing': '5'}}
    after = {'a.dcm': {**before['a.dcm'], 'time_of_processing': '2'}}
    assert compare_rows(before, after)['exact_medical_matches'] == 1
    after['a.dcm']['quality_prob'] = '.2'
    assert compare_rows(before, after)['exact_medical_matches'] == 0
    with pytest.raises(ValueError, match='different input'):
        compare_rows(before, {})


def test_duplicate_output_paths_are_rejected(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text('path_to_file,quality_class\na.dcm,0\na.dcm,0\n')
    with pytest.raises(ValueError, match='Duplicate output'):
        read_rows(path)
