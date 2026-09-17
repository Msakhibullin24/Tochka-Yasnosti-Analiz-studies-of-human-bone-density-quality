import io

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, ComputedRadiographyImageStorage, generate_uid
import pytest

from app.preprocessing import InvalidDicom, _normalize_pixels, prepare_dicom


def dicom_bytes(frames=1, rows=8, columns=8):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = ComputedRadiographyImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns, ds.NumberOfFrames = rows, columns, frames
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = np.arange(64, dtype=np.uint8).tobytes()
    stream = io.BytesIO()
    ds.save_as(stream, enforce_file_format=True)
    return stream.getvalue(), str(ds.SOPInstanceUID)


def test_rejects_multiframe_and_oversized_headers_before_decode(monkeypatch):
    def fail_decode(*args, **kwargs):
        raise AssertionError("Pixel decoder must not run for rejected headers")
    monkeypatch.setattr(pydicom.dataset.Dataset, "pixel_array", property(fail_decode))
    with pytest.raises(InvalidDicom, match="Multi-frame"):
        prepare_dicom(dicom_bytes(frames=2)[0])
    with pytest.raises(InvalidDicom, match="exceeds"):
        prepare_dicom(dicom_bytes(rows=5000, columns=5000)[0])


def test_valid_single_frame_preserves_image_uid_and_pixels():
    payload, uid = dicom_bytes()
    prepared = prepare_dicom(payload)
    assert prepared.image.shape == (8, 8)
    assert prepared.metadata["image_uid"] == uid
    with pytest.raises(InvalidDicom, match="exceeds"):
        prepare_dicom(payload, max_pixels=32)


def test_normalization_does_not_silently_take_first_frame_or_nan():
    with pytest.raises(InvalidDicom, match="single-frame"):
        _normalize_pixels(np.zeros((2, 8, 8)), "MONOCHROME2")
    with pytest.raises(InvalidDicom, match="non-finite"):
        _normalize_pixels(np.array([[0, 1], [2, np.nan]]), "MONOCHROME2")
