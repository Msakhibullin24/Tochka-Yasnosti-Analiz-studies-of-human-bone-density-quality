import csv
import warnings
import zipfile

import numpy as np
import pytest

from conftest import synthetic_spine, write_dicom
from dxaqc.dicom_io import DicomReadError, read_dxa
from dxaqc.geometry import measure_hip, measure_spine
from dxaqc.report import CONTRACT_COLUMNS, write_csv, write_xlsx

warnings.filterwarnings("ignore")


@pytest.mark.parametrize("angle", [-8.0, -3.0, 0.0, 4.0, 9.0])
def test_spine_axis_angle_is_measured_within_one_degree(angle):
    got = measure_spine(synthetic_spine(angle)).features["spine_angle_deg"]
    assert abs(got - angle) < 1.0


def test_iliac_crest_feature_separates_present_and_absent():
    full = synthetic_spine(0)
    cropped = full[:230]  # crests cut off, as in an incorrectly positioned scan
    assert measure_spine(full).features["crest_min_frac"] > 0.2
    assert measure_spine(cropped).features["crest_min_frac"] < 0.05


def test_hip_measurement_never_raises_on_degenerate_input():
    for img in (np.zeros((200, 200), np.uint8), np.full((200, 200), 255, np.uint8), synthetic_spine(0)):
        feats = measure_hip(img).features
        assert "shaft_angle_deg" in feats and "sat_frac" in feats


def test_reader_preserves_nonconformant_uid_and_scale(tmp_path):
    p = write_dicom(tmp_path / "a.dcm", synthetic_spine(0, 318, 300), exposed_area=(180, 193))
    img = read_dxa(p)
    assert img.study_uid.startswith("1.2.643.5.1.13.13.12.2.77.8252.0709")  # leading-zero component kept
    assert img.pixel_mm_source == "ExposedArea" and 0.59 < img.pixel_mm < 0.62


def test_reader_ignores_exposed_area_of_a_composite_study(tmp_path):
    img = read_dxa(write_dicom(tmp_path / "a.dcm", synthetic_spine(0, 263, 280), exposed_area=(520, 478)))
    assert img.pixel_mm_source == "device_default"


def test_reader_normalises_monochrome1_and_16bit(tmp_path):
    base = synthetic_spine(0)
    inv = read_dxa(write_dicom(tmp_path / "m1.dcm", 255 - base, photometric="MONOCHROME1"))
    assert np.abs(inv.pixels.astype(int) - base.astype(int)).max() <= 1
    wide = read_dxa(write_dicom(tmp_path / "w.dcm", base.astype(np.uint16) * 200))
    assert wide.pixels.dtype == np.uint8 and wide.pixels.max() > 200


@pytest.mark.parametrize("payload,code", [(b"not a dicom at all", "DICOM_PARSE_ERROR"), (b"", "DICOM_PARSE_ERROR")])
def test_reader_reports_garbage_as_coded_error(tmp_path, payload, code):
    p = tmp_path / "bad.dcm"
    p.write_bytes(payload)
    with pytest.raises(DicomReadError) as e:
        read_dxa(p)
    assert e.value.code in {code, "NO_PIXEL_DATA"}


def test_blank_image_is_a_failure_not_a_good_study(tmp_path):
    with pytest.raises(DicomReadError) as e:
        read_dxa(write_dicom(tmp_path / "blank.dcm", np.zeros((200, 200), np.uint8)))
    assert e.value.code == "EMPTY_IMAGE"


def test_report_contract_columns_come_first_and_xlsx_is_valid(tmp_path):
    rows = [{"path_to_study": "s", "study_uid": "1.2", "image_uid": "1.2.3", "anatomical_region": "Поясничный отдел позвоночника",
             "quality_class": 1, "violation_type": "Не выравнена ось позвоночника;Присутствуют посторонние предметы",
             "processing_status": "Success",
             "time_of_processing": 0.5, "quality_prob": 0.9, "violation_description": "<&> кириллица"}]
    write_csv(rows, tmp_path / "r.csv")
    header = next(csv.reader(open(tmp_path / "r.csv", encoding="utf-8-sig")))
    assert header[:8] == CONTRACT_COLUMNS
    write_csv(rows, tmp_path / "strict.csv", strict=True)
    assert next(csv.reader(open(tmp_path / "strict.csv", encoding="utf-8-sig"))) == CONTRACT_COLUMNS + ["quality_prob"]
    write_xlsx(rows, tmp_path / "r.xlsx")
    with zipfile.ZipFile(tmp_path / "r.xlsx") as z:
        sheet = z.read("xl/worksheets/sheet1.xml").decode()
    import xml.dom.minidom
    xml.dom.minidom.parseString(sheet)
    assert "Не выравнена ось позвоночника;Присутствуют посторонние предметы" in sheet and "&lt;&amp;&gt;" in sheet


def test_inverted_report_style_export_is_normalised_but_normal_frames_are_untouched(tmp_path):
    base = synthetic_spine(0)
    normal = read_dxa(write_dicom(tmp_path / "n.dcm", base))
    flipped = read_dxa(write_dicom(tmp_path / "i.dcm", 255 - base))  # stored as MONOCHROME2, drawn dark-on-bright
    assert "POLARITY_INVERTED" not in normal.warnings and "POLARITY_INVERTED" in flipped.warnings
    assert np.abs(flipped.pixels.astype(int) - normal.pixels.astype(int)).max() <= 1


def test_batch_walker_skips_non_dicom_payloads_but_keeps_extensionless_dicom(tmp_path):
    from dxaqc.pipeline import discover
    write_dicom(tmp_path / "IM0001", synthetic_spine(0))
    (tmp_path / "mask.seg.nrrd").write_bytes(b"NRRD0004\n")
    (tmp_path / "разметка.xlsx").write_bytes(b"PK")
    assert [p.name for p in discover(tmp_path)] == ["IM0001"]


def test_raster_images_from_public_datasets_can_be_read_for_training(tmp_path):
    import cv2
    from dxaqc.dicom_io import read_any
    folder = tmp_path / "папка"
    folder.mkdir()
    cv2.imencode(".png", synthetic_spine(3))[1].tofile(str(folder / "снимок.png"))
    img = read_any(folder / "снимок.png", pixel_mm=0.5)
    assert img.pixel_mm == 0.5 and img.pixel_mm_source == "labels_csv" and img.pixels.shape == (300, 300)
    assert abs(measure_spine(img.pixels, img.pixel_mm).features["spine_angle_deg"] - 3) < 1


def test_output_vocabulary_matches_the_organiser_closed_lists():
    from dxaqc.model import REGION_LABEL, VIOLATION_LABEL, official_violation_type
    assert set(REGION_LABEL.values()) == {"Поясничный отдел позвоночника", "Проксимальный отдел бедра"}
    spine = {VIOLATION_LABEL[c] for c in ("spine_coverage", "spine_axis", "spine_artifact")}
    hip = {VIOLATION_LABEL[c] for c in ("hip_position_rotation", "hip_roi_coverage", "hip_metal_implant")}
    assert spine == {"Некорректная укладка", "Не выравнена ось позвоночника", "Присутствуют посторонние предметы"}
    assert hip == {"Некорректная укладка", "Некорректная область интереса"}
    assert official_violation_type([]) == ""
    assert official_violation_type(["hip_roi_coverage", "hip_metal_implant"]) == "Некорректная область интереса"
    assert official_violation_type(["spine_axis", "spine_artifact"]) == "Не выравнена ось позвоночника;Присутствуют посторонние предметы"
