import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def synthetic_spine(angle_deg: float = 0.0, h: int = 300, w: int = 300) -> np.ndarray:
    """Bright vertical band (vertebral column) rotated by angle_deg, iliac blobs at the bottom."""
    import cv2
    img = np.zeros((h, w), np.uint8)
    cv2.rectangle(img, (w // 2 - 28, -50), (w // 2 + 28, h + 50), 170, -1)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)  # image y axis points down
    img = cv2.warpAffine(img, m, (w, h))
    cv2.circle(img, (25, h - 10), 45, 140, -1)
    cv2.circle(img, (w - 25, h - 10), 45, 140, -1)
    rng = np.random.default_rng(0)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 3).astype(np.int16) + rng.integers(0, 8, img.shape), 0, 255).astype(np.uint8)


def write_dicom(path, pixels: np.ndarray, *, photometric="MONOCHROME2", exposed_area=None,
                study_uid="1.2.643.5.1.13.13.12.2.77.8252.07090503010902110208051203150011",
                sop_uid="1.2.3.4.5.6.7.8.9.10"):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ImplicitVRLittleEndian
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, sop_uid
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, "1.2.3.4.5"
    ds.Modality, ds.Manufacturer = "CR", "GE Healthcare"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    bits = 8 if pixels.dtype == np.uint8 else 16
    ds.BitsAllocated = ds.BitsStored = bits
    ds.HighBit, ds.PixelRepresentation = bits - 1, 0
    if exposed_area:
        ds.add_new((0x0040, 0x0303), "US", list(exposed_area))
    ds.PixelData = pixels.tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


@pytest.fixture(scope="session")
def bundle_available():
    from dxaqc.pipeline import MODEL_PATH
    if not MODEL_PATH.exists():
        pytest.skip("models/bundle.joblib is not trained yet")
    return True


@pytest.fixture
def trusted_synthetic_router(monkeypatch):
    """Workflow tests use drawn phantoms, not an evaluation of router confidence.

    Preserve the region prediction but explicitly supply trusted routing. Scope rejection
    and organiser images are tested separately without this fixture.
    """
    from dxaqc.model import RegionRouter
    predict = RegionRouter.predict
    def trusted(self, embedding):
        regions, _ = predict(self, embedding)
        return regions, np.ones(len(regions))
    monkeypatch.setattr(RegionRouter, 'predict', trusted)
