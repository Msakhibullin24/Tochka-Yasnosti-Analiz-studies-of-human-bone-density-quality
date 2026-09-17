from pathlib import Path
import json
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, ComputedRadiographyImageStorage, generate_uid
import pytest

from app.competition_data import prepare, read_labels, pixel_fingerprint


def workbook(path: Path, *, bad_label: str | None = None) -> None:
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    root = ET.Element('worksheet', xmlns=ns)
    data = ET.SubElement(root, 'sheetData')
    rows = [
        {'B': 'study'},
        {'C': 'корректная укладка', 'D': 'правильно выравнена ось позвоночника',
         'E': 'наличие посторонних предметов', 'F': 'позиционирование/ротация',
         'G': 'корректности области интересов', 'H': 'позиционирование/ротация',
         'I': 'корректности области интересов', 'J': 'Позвоночник',
         'K': 'Проксимальный отдел правого бедра', 'L': 'Проксимальный отдел левого бедра'},
        {'B': 'study-a', 'C': bad_label or '0', 'D': '0', 'E': '0', 'J': '1'},
        {'B': 'study-b', 'F': '0', 'G': '0', 'K': '0'},
    ]
    for index, cells in enumerate(rows, 1):
        row = ET.SubElement(data, 'row', r=str(index))
        for col, value in cells.items():
            cell = ET.SubElement(row, 'c', r=f'{col}{index}', t='inlineStr')
            ET.SubElement(ET.SubElement(cell, 'is'), 't').text = value
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Калибровка" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml', ET.tostring(root))


def dicom(path: Path, pixels: np.ndarray, study_uid: str) -> None:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ImplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = ComputedRadiographyImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientID = 'Anonymized'
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)


def test_preparation_preserves_mapping_missing_labels_and_flags_duplicates(tmp_path):
    labels = tmp_path / 'labels.xlsx'
    workbook(labels)
    source = tmp_path / 'input'
    a = np.arange(120, dtype=np.uint8).reshape(10, 12)
    uid = generate_uid()
    dicom(source / 'study-a' / 'first.dcm', a, uid)
    dicom(source / 'study-a' / 'copy.dcm', a, uid)
    dicom(source / 'study-b' / 'cross-study.dcm', a, generate_uid())
    (source / 'study-b' / 'broken.dcm').write_bytes(b'broken')
    output = tmp_path / 'output'
    summary = prepare(source, labels, output)
    assert summary['source_file_count'] == 4
    assert summary['decoded_count'] == 3
    assert summary['unique_image_count'] == 2
    assert summary['unique_pixel_count'] == 1
    assert summary['duplicate_extra_files'] == 1
    assert summary['issue_counts']['cross_study_exact_duplicate'] == 1
    assert summary['issue_counts']['overall_criteria_disagreement'] == 1
    assert summary['issue_counts']['decode_failure'] == 1
    assert not summary['training_ready']
    records = [json.loads(line) for line in (output / 'manifest.jsonl').read_text().splitlines()]
    assert all(r['anatomical_region'] is None and r['quality_class'] is None for r in records)
    assert all((output / r['image_path']).is_file() for r in records)
    parsed = read_labels(labels)
    assert parsed[0]['regions']['hip_left']['quality_class'] is None
    mapped = [json.loads(line) for line in (output / 'study_labels.jsonl').read_text().splitlines()]
    assert mapped[0]['dicom_study_uids'] == [uid]
    assert all(r['candidate_fold'] is not None for r in records)
    second = tmp_path / 'again'
    assert prepare(source, labels, second) == summary
    assert (second / 'manifest.jsonl').read_bytes() == (output / 'manifest.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        prepare(source, labels, output)
    assert (output.stat().st_mode & 0o777) == 0o700


def test_bad_label_and_source_are_rejected_without_partial_export(tmp_path):
    labels = tmp_path / 'labels.xlsx'
    workbook(labels, bad_label='2')
    with pytest.raises(ValueError, match='Non-binary'):
        read_labels(labels)
    workbook(labels)
    source = tmp_path / 'input'
    source.mkdir()
    with pytest.raises(ValueError, match='No DICOM'):
        prepare(source, labels, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
    assert not list(tmp_path.glob('.competition-*'))
    with pytest.raises(ValueError, match='outside source'):
        prepare(source, labels, source / 'output')


def test_pixel_hash_includes_geometry_and_photometric():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert pixel_fingerprint(a, 'MONOCHROME2') != pixel_fingerprint(a.reshape(4, 3), 'MONOCHROME2')
    assert pixel_fingerprint(a, 'MONOCHROME2') != pixel_fingerprint(a, 'MONOCHROME1')
