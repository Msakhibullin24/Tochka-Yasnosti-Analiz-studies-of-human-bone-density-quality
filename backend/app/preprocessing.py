from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any


class InvalidDicom(ValueError):
    pass


class ExcludedDicom(ValueError):
    """A readable presentation object that must not enter the ML dataset."""

    def __init__(self, message: str, *, reason: str = "secondary_capture"):
        super().__init__(message)
        self.reason = reason


@dataclass
class PreparedStudy:
    image: Any
    width: int
    height: int
    preview_url: str
    content_bbox: tuple[float, float, float, float] | None
    metadata: dict[str, Any]


def _safe_text(dataset, keyword: str, default: str = "") -> str:
    value = getattr(dataset, keyword, default)
    if value is None:
        return default
    return str(value).replace("\x00", "").strip()[:256]


def _content_bbox(image) -> tuple[float, float, float, float] | None:
    import numpy as np

    low, high = np.percentile(image, (5, 99))
    if high <= low:
        return None
    mask = image > (low + (high - low) * 0.08)
    ys, xs = np.where(mask)
    if xs.size < image.size * 0.01:
        return None
    return (
        float(xs.min() / max(1, image.shape[1] - 1)),
        float(ys.min() / max(1, image.shape[0] - 1)),
        float(xs.max() / max(1, image.shape[1] - 1)),
        float(ys.max() / max(1, image.shape[0] - 1)),
    )


def _normalize_pixels(pixel_array, photometric: str):
    import numpy as np

    image = np.asarray(pixel_array)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        rgb = image[..., :3].astype(np.float32)
        image = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    elif image.ndim > 2:
        image = image[0]
    if image.ndim != 2:
        raise InvalidDicom("Only single-frame grayscale DXA images are supported")
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise InvalidDicom("Pixel Data contains no finite values")
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        raise InvalidDicom("Pixel Data has no usable intensity range")
    image = np.clip((image - low) / (high - low), 0, 1)
    if photometric.upper() == "MONOCHROME1":
        image = 1 - image
    return (image * 255).round().astype(np.uint8)


def prepare_dicom(payload: bytes) -> PreparedStudy:
    try:
        import pydicom
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("DICOM runtime is not installed; install backend dependencies") from error

    try:
        dataset = pydicom.dcmread(io.BytesIO(payload), force=False)
        sop_class_uid = _safe_text(dataset, "SOPClassUID")
        samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
        photometric = _safe_text(dataset, "PhotometricInterpretation", "MONOCHROME2")
        presentation_rgb = samples_per_pixel > 1 or photometric.upper().startswith(("RGB", "YBR"))
        if presentation_rgb:
            raise ExcludedDicom(
                "Presentation RGB DICOM is readable but excluded from ML training and anatomical inference",
                reason="secondary_capture" if sop_class_uid == "1.2.840.10008.5.1.4.1.1.7" else "color_presentation",
            )
        raw_pixels = dataset.pixel_array
    except ExcludedDicom:
        raise
    except Exception as error:
        raise InvalidDicom(f"Unable to decode DICOM Pixel Data: {error}") from error

    image = _normalize_pixels(raw_pixels, _safe_text(dataset, "PhotometricInterpretation", "MONOCHROME2"))
    height, width = image.shape
    preview = Image.fromarray(image, mode="L")
    preview.thumbnail((1024, 1024))
    preview_buffer = io.BytesIO()
    preview.save(preview_buffer, format="PNG", optimize=True)
    preview_url = "data:image/png;base64," + base64.b64encode(preview_buffer.getvalue()).decode("ascii")

    metadata = {
        "patient_id": _safe_text(dataset, "PatientID"),
        "accession_number": _safe_text(dataset, "AccessionNumber"),
        "study_uid": _safe_text(dataset, "StudyInstanceUID"),
        "series_uid": _safe_text(dataset, "SeriesInstanceUID"),
        "study_date": _safe_text(dataset, "StudyDate"),
        "study_time": _safe_text(dataset, "StudyTime"),
        "manufacturer": _safe_text(dataset, "Manufacturer"),
        "model_name": _safe_text(dataset, "ManufacturerModelName"),
        "institution_present": bool(_safe_text(dataset, "InstitutionName")),
        "operator_present": bool(_safe_text(dataset, "OperatorsName")),
        "modality": _safe_text(dataset, "Modality", "OT"),
        "body_part": _safe_text(dataset, "BodyPartExamined"),
        "description": " ".join(filter(None, (_safe_text(dataset, "StudyDescription"), _safe_text(dataset, "SeriesDescription")))),
        "protocol_name": _safe_text(dataset, "ProtocolName"),
        "pixel_spacing": " × ".join(str(item) for item in getattr(dataset, "PixelSpacing", [])),
        "photometric": _safe_text(dataset, "PhotometricInterpretation"),
        "transfer_syntax_uid": str(getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", "")),
        "bits_allocated": int(getattr(dataset, "BitsAllocated", 0) or 0),
        "sop_class_uid": _safe_text(dataset, "SOPClassUID"),
        "samples_per_pixel": int(getattr(dataset, "SamplesPerPixel", 1) or 1),
        "training_eligible": True,
        "patient_identity_removed": _safe_text(dataset, "PatientIdentityRemoved").upper() == "YES",
        "burned_in_annotation": _safe_text(dataset, "BurnedInAnnotation").upper() or None,
    }
    return PreparedStudy(image, width, height, preview_url, _content_bbox(image), metadata)
