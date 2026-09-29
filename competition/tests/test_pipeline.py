from dxaqc.report import CONTRACT_COLUMNS
import csv
import warnings
import zipfile

import pytest

from conftest import synthetic_spine, write_dicom
from dxaqc.pipeline import Options, assign_study_paths, run_batch, safe_extract

warnings.filterwarnings("ignore")


def _rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


@pytest.fixture()
def study_dir(tmp_path):
    root = tmp_path / "in"
    (root / "study_A" / "series").mkdir(parents=True)
    (root / "study_B").mkdir()
    write_dicom(root / "study_A" / "series" / "CR000000.dcm", synthetic_spine(0), sop_uid="1.1", study_uid="1.2.826.1")
    write_dicom(root / "study_A" / "series" / "CR000001.dcm", synthetic_spine(0), sop_uid="1.2", study_uid="1.2.826.1")  # exact pixel copy
    write_dicom(root / "study_B" / "CR000000.dcm", synthetic_spine(9), sop_uid="2.1")
    (root / "study_B" / "broken.dcm").write_bytes(b"\x00" * 300)
    (root / "study_B" / "разметка.xlsx").write_bytes(b"ignored")
    return root


def test_every_file_gets_a_row_and_failures_do_not_stop_the_batch(study_dir, tmp_path, bundle_available, trusted_synthetic_router):
    summary = run_batch(study_dir, tmp_path / "out", Options(explanations=True))
    rows = _rows(tmp_path / "out" / "results.csv")
    assert summary["files"] == len(rows) == 4 and summary["failure"] == 1
    by_uid = {r["image_uid"]: r for r in rows}
    assert by_uid["1.2"]["duplicate_of"] == "1.1"
    assert by_uid["1.2"]["quality_class"] == by_uid["1.1"]["quality_class"]
    bad = next(r for r in rows if r["processing_status"] == "Failure")
    assert bad["quality_class"] == "" and bad["quality_prob"] == "" and bad["error_code"] and bad["violation_type"] == ""
    allowed = {"", "Некорректная укладка", "Не выравнена ось позвоночника", "Присутствуют посторонние предметы",
               "Некорректная область интереса"}
    for r in rows:
        assert set(r["violation_type"].split(";")) <= allowed
        assert (r["violation_type"] == "") or r["quality_class"] == "1"
        if r["processing_status"] == "Success":
            assert 0.0 <= float(r["quality_prob"]) <= 1.0
    assert {r["path_to_study"] for r in rows if r["processing_status"] == "Success"} == {"study_A", "study_B"}
    assert all(r["anatomical_region"] for r in rows if r["processing_status"] == "Success")
    assert (tmp_path / "out" / "results.xlsx").exists() and (tmp_path / "out" / "additional_series.zip").exists()
    with zipfile.ZipFile(tmp_path / "out" / "additional_series.zip") as z:
        names = z.namelist()
    assert any(n.endswith("_qc_sr.dcm") for n in names) and any(n.endswith("_qc_sc.dcm") for n in names)


def test_results_are_reproducible_and_zip_input_equals_folder_input(study_dir, tmp_path, bundle_available):
    archive = tmp_path / "in.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in sorted(study_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(study_dir).as_posix())
    outs = []
    for name, src in (("a", study_dir), ("b", study_dir), ("z", archive)):
        run_batch(src, tmp_path / name, Options(explanations=False))
        outs.append([{k: v for k, v in r.items() if k != "time_of_processing"} for r in _rows(tmp_path / name / "results.csv")])
    assert outs[0] == outs[1] == outs[2]


def test_wrapper_folder_is_not_reported_as_the_study(study_dir, tmp_path, bundle_available):
    wrapped = tmp_path / "wrapped" / "Исследования"
    wrapped.parent.mkdir()
    study_dir.rename(wrapped)
    run_batch(wrapped.parent, tmp_path / "out", Options(explanations=False))
    assert {r["path_to_study"] for r in _rows(tmp_path / "out" / "results.csv")} >= {"Исследования/study_A", "Исследования/study_B"}


def test_study_paths_use_uid_unique_folder_across_wrappers_and_duplicate_exports():
    paths = [
        ('НД_для_обучения/Исследования/a/series/images/1.dcm', 'a'),
        ('НД_для_обучения/Исследования/b/series/images/2.dcm', 'b'),
        ('Для теста/3.dcm', 'c'),
        ('Для теста/4.dcm', 'd'),
        ('Annotation/00001/images/hip.dcm', 'e'),
        ('Annotation/00002/images/hip.dcm', 'f'),
        ('Verification/00001/images/hip.dcm', 'e'),
        ('Verification/00002/images/hip.dcm', 'f'),
        ('Verification/00002/images/broken.dcm', ''),
    ]
    rows = [{'path_to_file': path, 'study_uid': uid} for path, uid in paths]
    assign_study_paths(rows)
    assert [r['path_to_study'] for r in rows] == [
        'НД_для_обучения/Исследования/a',
        'НД_для_обучения/Исследования/b',
        'Для теста/3.dcm', 'Для теста/4.dcm',
        'Annotation/00001', 'Annotation/00002', 'Verification/00001',
        'Verification/00002', 'Verification/00002',
    ]


def test_duplicates_have_source_linked_explanations_and_cli_keeps_only_the_zip(study_dir, tmp_path, bundle_available):
    run_batch(study_dir, tmp_path / "out", Options(explanations=True))
    rows = {r["image_uid"]: r for r in _rows(tmp_path / "out" / "results.csv")}
    assert rows["1.2"]["explanation_png"] != rows["1.1"]["explanation_png"]
    assert rows["1.2"]["explanation_png"] and rows["1.1"]["explanation_png"]
    assert not (tmp_path / "out" / "additional_series").exists()
    with zipfile.ZipFile(tmp_path / "out" / "additional_series.zip") as z:
        assert rows["1.1"]["explanation_png"] in z.namelist()


def test_hostile_archive_still_produces_a_table(study_dir, tmp_path, bundle_available, trusted_synthetic_router):
    inner = tmp_path / "study.zip"
    with zipfile.ZipFile(inner, "w") as z:
        z.write(study_dir / "study_B" / "CR000000.dcm", "deep/CR000000.dcm")
    archive = tmp_path / "hostile.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(inner, "study.zip")
        z.writestr("study", "a plain file that collides with the nested archive name")
        z.writestr("broken.zip", b"PK\x03\x04 truncated")
        z.writestr("huge.dcm", b"")
        from dxaqc.report import write_xlsx
        write_xlsx([{"path_to_study": "x"}], tmp_path / "labels.xlsx")
        z.write(tmp_path / "labels.xlsx", "разметка.xlsx")  # an office file is a zip container: must be ignored
    summary = run_batch(archive, tmp_path / "out", Options(explanations=False))
    rows = _rows(tmp_path / "out" / "results.csv")
    assert summary["success"] == 1 and len(rows) == summary["files"] >= 3
    assert not any("xlsx" in r["path_to_file"] for r in rows)


def test_oversized_header_is_rejected_before_decoding(tmp_path):
    import numpy as np
    import pydicom
    from dxaqc.dicom_io import DicomReadError, read_dxa
    p = write_dicom(tmp_path / "bomb.dcm", synthetic_spine(0))
    ds = pydicom.dcmread(p)
    ds.Rows, ds.Columns = 60000, 60000
    ds.save_as(p)
    with pytest.raises(DicomReadError) as e:
        read_dxa(p)
    assert e.value.code == "UNSUPPORTED_IMAGE_SIZE"


def test_cli_writes_a_failure_table_when_the_input_is_missing(tmp_path):
    from dxaqc.cli import main
    assert main(["--input", str(tmp_path / "nope"), "--output", str(tmp_path / "out")]) == 2
    rows = _rows(tmp_path / "out" / "results.csv")
    assert rows[0]["processing_status"] == "Failure"
    assert list(rows[0]) == CONTRACT_COLUMNS + ["quality_prob"]
    extended = _rows(tmp_path / "out" / "results_extended.csv")
    assert extended[0]["error_code"] == "BATCH_INPUT_ERROR"


def test_cli_preserves_previous_run_and_rejects_output_inside_input(tmp_path):
    from dxaqc.cli import main
    previous = tmp_path / 'out'
    previous.mkdir()
    (previous / 'results.csv').write_text('previous-result')
    (previous / 'additional_series.zip').write_bytes(b'previous-series')
    assert main(['--input', str(tmp_path / 'missing'), '--output', str(previous)]) == 2
    assert (previous / 'results.csv').read_text() == 'previous-result'
    assert (previous / 'additional_series.zip').read_bytes() == b'previous-series'
    source = tmp_path / 'source'
    source.mkdir()
    assert main(['--input', str(source), '--output', str(source / 'out')]) == 2
    assert not (source / 'out').exists()


def test_zip_slip_entries_are_not_written_outside(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../../escaped.txt", "x")
        z.writestr("ok/file.dcm", "y")
    safe_extract(archive, tmp_path / "dest")
    assert not (tmp_path / "escaped.txt").exists() and not (tmp_path.parent / "escaped.txt").exists()
    assert (tmp_path / "dest" / "ok" / "file.dcm").exists()


def test_api_batch_roundtrip(study_dir, tmp_path, bundle_available, monkeypatch):
    monkeypatch.setenv("DXAQC_DATA_DIR", str(tmp_path / "jobs"))
    import importlib
    import dxaqc.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    with TestClient(api.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        files = [("files", (p.name, p.read_bytes(), "application/dicom")) for p in sorted(study_dir.rglob("*.dcm"))[:2]]
        job = client.post("/api/v1/batch?wait=true", files=files).json()
        assert job["status"] == "finished" and job["summary"]["files"] == 2
        assert len(client.get(f"/api/v1/jobs/{job['id']}/rows").json()) == 2
        assert client.get(f"/api/v1/jobs/{job['id']}/results.csv").status_code == 200
        submission = client.get(f"/api/v1/jobs/{job['id']}/submission.csv")
        assert submission.status_code == 200
        assert len(next(csv.reader(submission.text.lstrip("\ufeff").splitlines()))) == 9
        acceptance = client.get(f"/api/v1/jobs/{job['id']}/submission_validation.json")
        assert acceptance.status_code == 200 and acceptance.json()['profile'] == 'competition_v2'
        png = next(r["explanation_png"] for r in client.get(f"/api/v1/jobs/{job['id']}/rows").json() if r["explanation_png"])
        assert client.get(f"/api/v1/jobs/{job['id']}/overlay", params={"path": png}).headers["content-type"] == "image/png"
        assert client.get(f"/api/v1/jobs/{job['id']}/overlay", params={"path": "../results.csv"}).status_code == 404
        assert client.get(f"/api/v1/jobs/{job['id']}/../../etc/passwd").status_code in (404, 422)
        bad = client.post("/api/v1/analyze", files={"file": ("x.dcm", b"garbage", "application/dicom")})
        assert bad.status_code == 422 and bad.json()["processing_status"] == "Failure"


DEBUG_SET = __import__("pathlib").Path(__import__("os").environ.get("DXAQC_DEBUG_SET", "/nonexistent"))


@pytest.mark.skipif(not DEBUG_SET.exists(), reason="organiser debug set is not mounted (set DXAQC_DEBUG_SET)")
def test_organiser_debug_set_regions(tmp_path, bundle_available):
    run_batch(DEBUG_SET, tmp_path / "out", Options(explanations=False))
    got = {r["path_to_file"]: (r["anatomical_region"], r["laterality"]) for r in _rows(tmp_path / "out" / "results.csv")}
    expected = {"ПОП": ("Поясничный отдел позвоночника", ""), "ППОБ": ("Проксимальный отдел бедра", "right"),
                "ЛПОБ": ("Проксимальный отдел бедра", "left")}
    for name, region in got.items():
        for tag, want in expected.items():
            if f"_{tag}." in name:
                assert region == want


def test_cache_respects_spacing_and_input_order(tmp_path, bundle_available, trusted_synthetic_router):
    import json
    import pydicom
    root = tmp_path / 'in'
    root.mkdir()
    paths = []
    for i, mm in enumerate((0.4, 1.2)):
        p = write_dicom(root / f'{i}.dcm', synthetic_spine(4), study_uid=f'1.2.{i}', sop_uid=f'1.3.{i}')
        ds = pydicom.dcmread(p)
        ds.PixelSpacing = [mm, mm]
        ds.save_as(p)
        paths.append(p)
    run_batch(root, tmp_path / 'out', Options(explanations=False))
    rows = _rows(tmp_path / 'out/results.csv')
    assert [json.loads(r['measurements'])['image_height_mm'] for r in rows] == [120, 360]
    assert not any(r['duplicate_of'] for r in rows)
    paths[0].rename(root / 'last.dcm')
    run_batch(root, tmp_path / 'out2', Options(explanations=False))
    by_uid = lambda rs: {r['image_uid']: (r['quality_class'], r['quality_prob'], r['measurements']) for r in rs}
    assert by_uid(rows) == by_uid(_rows(tmp_path / 'out2/results.csv'))


def test_broken_nested_zip_is_reported(tmp_path, bundle_available):
    archive = tmp_path / 'in.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('nested.zip', b'broken')
    summary = run_batch(archive, tmp_path / 'out', Options(explanations=False))
    assert summary['failure'] == 1
    assert _rows(tmp_path / 'out/results.csv')[0]['error_code'] == 'ARCHIVE_INPUT_ERROR'


def test_archive_budget_failure_writes_csv_and_xlsx(tmp_path, monkeypatch):
    import dxaqc.pipeline as pipeline
    monkeypatch.setattr(pipeline, 'MAX_ARCHIVE_BYTES', 1)
    archive = tmp_path / 'in.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('a.dcm', b'1234')
    with pytest.raises(ValueError):
        run_batch(archive, tmp_path / 'out')
    assert _rows(tmp_path / 'out/results.csv')[0]['error_code'] == 'BATCH_INPUT_ERROR'
    assert (tmp_path / 'out/results.xlsx').exists()


def test_cancellation_keeps_completed_rows_and_accounts_for_remainder(study_dir, tmp_path, bundle_available):
    done = []
    summary = run_batch(study_dir, tmp_path / 'out', Options(explanations=False),
                        progress=lambda i, n: done.append(i), cancelled=lambda: bool(done))
    rows = _rows(tmp_path / 'out/results.csv')
    assert len(rows) == 4 and rows[0]['processing_status'] == 'Success'
    assert all(r['error_code'] == 'CANCELLED' for r in rows[1:])
    assert summary['cancelled'] is True


def test_personal_demographics_do_not_change_machine_decision(tmp_path, bundle_available, trusted_synthetic_router):
    import pydicom
    import numpy as np
    from dxaqc.pipeline import Analyzer
    from dxaqc.dicom_io import read_any
    path = write_dicom(tmp_path / 'a.dcm', synthetic_spine(7))
    analyzer = Analyzer()
    baseline = analyzer.analyze(read_any(path))
    ds = pydicom.dcmread(path)
    ds.PatientName = 'Synthetic^Patient'
    ds.PatientID = 'test-only'
    ds.PatientBirthDate = '19800101'
    ds.PatientSex = 'F'
    ds.PatientAge = '046Y'
    ds.save_as(path, enforce_file_format=True)
    changed = analyzer.analyze(read_any(path))
    for key in baseline:
        if key == 'features':
            assert baseline[key].keys() == changed[key].keys()
            np.testing.assert_allclose(list(baseline[key].values()), list(changed[key].values()), rtol=0, atol=0, equal_nan=True)
        elif key == 'overlay':
            np.testing.assert_array_equal(baseline[key], changed[key])
        else:
            assert baseline[key] == changed[key], key
