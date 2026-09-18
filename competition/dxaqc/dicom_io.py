"""Robust DICOM reading for DXA quality control.

Design rules (from the dataset audit):
* UIDs in the organiser export violate the DICOM UID syntax (leading zeros, >64 chars).
  They are passed through verbatim - never validated, never rewritten.
* No personal data is required; only technical tags are read.
* PixelSpacing is absent. The physical scale is recovered from ExposedArea (0040,0303)
  when it describes this image, otherwise the device constant is used.
* Any failure is turned into DicomReadError with a short machine-readable code.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import DEFAULT_PIXEL_MM

MAX_PIXELS = 6000 * 6000


class DicomReadError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class DxaImage:
    path: str
    pixels: np.ndarray  # uint8, bone bright
    study_uid: str
    series_uid: str
    image_uid: str
    pixel_mm: float
    pixel_mm_source: str
    pixel_sha256: str
    manufacturer: str = ""
    model_name: str = ""
    warnings: list[str] = field(default_factory=list)


def _text(ds, key: str) -> str:
    try:
        v = ds.get(key, "")
        return "" if v is None else str(v).replace("\x00", "").strip()
    except Exception:  # malformed element
        return ""


def _to_uint8(arr: np.ndarray, photometric: str, ds) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] in (3, 4):  # RGB secondary capture -> luminance
        a = a[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    elif a.ndim == 3:  # multi-frame: first frame
        a = a[0]
    if a.ndim != 2:
        raise DicomReadError("UNSUPPORTED_PIXEL_LAYOUT", f"unsupported pixel array shape {a.shape}")
    if a.size == 0 or a.size > MAX_PIXELS:
        raise DicomReadError("UNSUPPORTED_IMAGE_SIZE", f"image size {a.shape} is out of range")
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        slope, inter = float(getattr(ds, "RescaleSlope", 1) or 1), float(getattr(ds, "RescaleIntercept", 0) or 0)
        a = a * slope + inter
        lo, hi = np.percentile(a, (0.5, 99.5))
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
        a = a.astype(np.uint8)
    if photometric == "MONOCHROME1":
        a = 255 - a
    return np.ascontiguousarray(a)


def normalise_polarity(pixels: np.ndarray) -> tuple[np.ndarray, bool]:
    """Guarantee 'bone bright on a dark background' whatever the exporter did.

    Report-style exports (e.g. Hologic screen captures) are stored as MONOCHROME2 but drawn dark-on-bright.
    The image border of a DXA frame is air/soft tissue, i.e. background: if it is bright, the frame is inverted.
    """
    h, w = pixels.shape
    m = max(2, min(h, w) // 20)
    border = np.concatenate([pixels[:m].ravel(), pixels[-m:].ravel(), pixels[:, :m].ravel(), pixels[:, -m:].ravel()])
    if float(np.median(border)) > 140 and float(np.median(pixels)) > 128:
        return np.ascontiguousarray(255 - pixels), True
    return pixels, False


def looks_like_dicom(path: str | Path) -> bool:
    """Cheap sniff used by the batch walker: DICM magic, or a DICOM-ish name for preamble-less files."""
    p = Path(path)
    try:
        with open(p, "rb") as f:
            head = f.read(132)
    except OSError:
        return False
    if head[128:132] == b"DICM":
        return True
    return p.suffix.lower() in {"", ".dcm", ".dicom", ".ima", ".img"} or p.suffix[1:].isdigit()


def _pixel_mm(ds, rows: int, cols: int) -> tuple[float, str]:
    for key in ("PixelSpacing", "ImagerPixelSpacing"):
        v = ds.get(key)
        try:
            if v is not None and len(v) == 2 and 0.05 < float(v[0]) < 5:
                return float(v[0]), key
        except Exception:
            pass
    try:
        ea = ds.get((0x0040, 0x0303))
        if ea is not None:
            width_mm, length_mm = float(ea.value[0]), float(ea.value[1])
            if width_mm > 0 and length_mm > 0:
                sx, sy = width_mm / cols, length_mm / rows
                # Accept only when the tag describes this image (isotropic, plausible).
                if abs(sx - sy) / max(sx, sy) < 0.06 and 0.3 < sy < 1.2:
                    return (sx + sy) / 2, "ExposedArea"
    except Exception:
        pass
    return DEFAULT_PIXEL_MM, "device_default"


def read_dxa(path: str | Path) -> DxaImage:
    import pydicom
    from pydicom.errors import InvalidDicomError

    path = Path(path)
    notes: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                ds = pydicom.dcmread(str(path))
            except InvalidDicomError:
                ds = pydicom.dcmread(str(path), force=True)
                notes.append("NO_DICOM_PREAMBLE")
            if any("Invalid value for VR UI" in str(w.message) for w in caught):
                notes.append("NON_CONFORMANT_UID_PRESERVED")
    except Exception as exc:
        raise DicomReadError("DICOM_PARSE_ERROR", f"{type(exc).__name__}: {exc}") from exc
    if "PixelData" not in ds:
        raise DicomReadError("NO_PIXEL_DATA", "DICOM object has no PixelData (report/SR/presentation state)")
    try:
        declared = int(ds.get("Rows", 0) or 0) * int(ds.get("Columns", 0) or 0) * max(int(ds.get("NumberOfFrames", 1) or 1), 1)
    except Exception:
        declared = 0
    if declared > MAX_PIXELS:  # checked on the header: decoding a bomb would be killed by the OOM killer
        raise DicomReadError("UNSUPPORTED_IMAGE_SIZE", f"declared pixel count {declared} exceeds the limit")
    if not hasattr(ds.file_meta, "TransferSyntaxUID"):
        from pydicom.uid import ImplicitVRLittleEndian
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr = ds.pixel_array
    except MemoryError as exc:
        raise DicomReadError("UNSUPPORTED_IMAGE_SIZE", "not enough memory to decode the image") from exc
    except Exception as exc:
        raise DicomReadError("PIXEL_DECODE_ERROR", f"{type(exc).__name__}: {exc}") from exc
    photometric = _text(ds, "PhotometricInterpretation").upper()
    pixels = _to_uint8(arr, photometric, ds)
    if min(pixels.shape) < 64:
        raise DicomReadError("UNSUPPORTED_IMAGE_SIZE", f"image {pixels.shape} is too small for analysis")
    if int(pixels.max()) - int(pixels.min()) < 10:
        raise DicomReadError("EMPTY_IMAGE", "image has no contrast")
    pixels, inverted = normalise_polarity(pixels)
    if inverted:
        notes.append("POLARITY_INVERTED")
    mm, src = _pixel_mm(ds, *pixels.shape)
    return DxaImage(
        path=str(path),
        pixels=pixels,
        study_uid=_text(ds, "StudyInstanceUID"),
        series_uid=_text(ds, "SeriesInstanceUID"),
        image_uid=_text(ds, "SOPInstanceUID"),
        pixel_mm=mm,
        pixel_mm_source=src,
        pixel_sha256=hashlib.sha256(pixels.tobytes() + str(pixels.shape).encode()).hexdigest(),
        manufacturer=_text(ds, "Manufacturer"),
        model_name=_text(ds, "ManufacturerModelName"),
        warnings=notes,
    )


RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_any(path: str | Path, pixel_mm: float | None = None) -> DxaImage:
    """Training-time loader: DICOM, or a raster image from a public dataset (PNG/JPG...).

    Rasters carry no UIDs and no physical scale: pass pixel_mm from the dataset documentation,
    otherwise the device default is assumed and mm-based measurements are only approximate.
    """
    path = Path(path)
    if path.suffix.lower() not in RASTER_SUFFIXES:
        return read_dxa(path)
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)  # np.fromfile: cv2.imread cannot open non-ASCII paths
    arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if arr is None:
        raise DicomReadError("PIXEL_DECODE_ERROR", "raster image cannot be decoded")

    class _NoTags:  # rasters have no rescale tags
        pass
    pixels = _to_uint8(arr, "MONOCHROME2", _NoTags())
    if min(pixels.shape) < 64 or int(pixels.max()) - int(pixels.min()) < 10:
        raise DicomReadError("EMPTY_IMAGE", "image is too small or has no contrast")
    pixels, inverted = normalise_polarity(pixels)
    return DxaImage(path=str(path), pixels=pixels, study_uid="", series_uid="", image_uid="",
                    pixel_mm=float(pixel_mm) if pixel_mm and pixel_mm == pixel_mm else DEFAULT_PIXEL_MM,
                    pixel_mm_source="labels_csv" if pixel_mm and pixel_mm == pixel_mm else "device_default",
                    pixel_sha256=hashlib.sha256(pixels.tobytes() + str(pixels.shape).encode()).hexdigest(),
                    warnings=["POLARITY_INVERTED"] if inverted else [])


def read_identifiers(path: str | Path) -> tuple[str, str]:
    """Best-effort UIDs for the report row of a file whose pixels cannot be read."""
    try:
        import pydicom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return _text(ds, "StudyInstanceUID"), _text(ds, "SOPInstanceUID")
    except Exception:
        return "", ""
