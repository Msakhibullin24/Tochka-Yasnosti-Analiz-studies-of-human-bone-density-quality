"""Visual and textual explanations: overlay image, DICOM Secondary Capture series, DICOM SR.

All generated UIDs are derived from the source SOPInstanceUID (2.25.<hash>), so repeated runs
produce identical objects. No patient attributes are copied except the (already anonymised)
study linkage needed to attach the objects to the source study.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .model import VIOLATION_RU

OK, BAD, INFO = (80, 220, 80), (60, 60, 255), (255, 200, 0)  # BGR


def derived_uid(source_uid: str, purpose: str) -> str:
    digest = hashlib.sha256(f"{purpose}|{source_uid}".encode()).hexdigest()
    return "2.25." + str(int(digest[:30], 16))


def _pt(p, scale, mirror_w=None):
    x, y = p
    if mirror_w is not None:
        x = mirror_w - 1 - x
    return int(round(x * scale)), int(round(y * scale))


def render_overlay(pixels: np.ndarray, region: str, overlay: dict, features: dict, violations: list[str],
                   quality_class: int, score: float, pixel_mm: float) -> np.ndarray:
    """Return a BGR image (upscaled x2) with landmarks, measured values and the verdict."""
    scale = 2.0
    h, w = pixels.shape
    canvas = cv2.cvtColor(cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    mirror = w if region == "hip_left" else None  # hip landmarks were measured on the mirrored image
    lines: list[str] = []
    if region == "spine":
        bad_axis = "spine_axis" in violations
        cl = [_pt(p, scale) for p in overlay.get("centerline", [])]
        for a, b in zip(cl[:-1], cl[1:]):
            cv2.line(canvas, a, b, INFO, 1, cv2.LINE_AA)
        ax = [_pt(p, scale) for p in overlay.get("axis", [])]
        if len(ax) == 2:
            cv2.line(canvas, ax[0], ax[1], BAD if bad_axis else OK, 2, cv2.LINE_AA)
            cv2.line(canvas, (ax[0][0], ax[0][1]), (ax[0][0], ax[1][1]), (200, 200, 200), 1, cv2.LINE_AA)
        lines.append(f"axis tilt {features.get('spine_angle_deg', float('nan')):+.1f} deg (limit 5)")
        lines.append(f"iliac crest visible: L {features.get('crest_left_frac', 0):.2f} R {features.get('crest_right_frac', 0):.2f}")
        if "spine_artifact" in violations:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            th = cv2.morphologyEx(pixels, cv2.MORPH_TOPHAT, kernel)
            mask = ((th > 60) | (pixels >= 250)).astype(np.uint8)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
            mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, [c for c in contours if cv2.contourArea(c) >= 12], -1, BAD, 1)
    else:
        sa = [_pt(p, scale, mirror) for p in overlay.get("shaft_axis", [])]
        if len(sa) == 2:
            cv2.line(canvas, sa[0], sa[1], INFO, 2, cv2.LINE_AA)
        for key, label in (("lesser_trochanter", "LT"), ("greater_trochanter_top", "GT"), ("ischium_bottom", "IS")):
            for p in overlay.get(key, []):
                q = _pt(p, scale, mirror)
                cv2.circle(canvas, q, 6, BAD if violations else OK, 2, cv2.LINE_AA)
                cv2.putText(canvas, label, (q[0] + 8, q[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INFO, 1, cv2.LINE_AA)
        # Required margins: 3 cm top/bottom, 2 cm lateral.
        m3, m2 = int(30 / pixel_mm * scale), int(20 / pixel_mm * scale)
        H, W = canvas.shape[:2]
        cv2.line(canvas, (0, m3), (W, m3), (160, 160, 160), 1)
        cv2.line(canvas, (0, H - m3), (W, H - m3), (160, 160, 160), 1)
        xl = m2 if region == "hip_right" else W - m2
        cv2.line(canvas, (xl, 0), (xl, H), (160, 160, 160), 1)
        lines.append(f"shaft angle {features.get('shaft_angle_deg', float('nan')):+.1f} deg")
        lines.append(f"lesser trochanter {features.get('lesser_troch_protrusion_mm', float('nan')):.1f} mm")
        lines.append(f"margins top {features.get('troch_top_margin_mm', float('nan')):.0f} / bottom "
                     f"{features.get('ischium_bottom_margin_mm', float('nan')):.0f} / lateral "
                     f"{features.get('lateral_margin_mm', float('nan')):.0f} mm")
    verdict = f"QC: {'VIOLATION' if quality_class else 'OK'}  score={score:.2f}"
    banner = np.zeros((22 * (len(lines) + 1 + len(violations)) + 8, canvas.shape[1], 3), np.uint8)
    y = 18
    cv2.putText(banner, verdict, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BAD if quality_class else OK, 1, cv2.LINE_AA)
    for v in violations:
        y += 22
        cv2.putText(banner, "- " + v, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BAD, 1, cv2.LINE_AA)
    for t in lines:
        y += 22
        cv2.putText(banner, t, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([canvas, banner])


def _base_dataset(sop_class: str, sop_uid: str, study_uid: str, series_uid: str, modality: str):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = "2.25.330164283640182934672119047201"
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, sop_uid
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
    ds.Modality = modality
    ds.PatientName, ds.PatientID = "Anonymized", "Anonymized"
    ds.Manufacturer, ds.ManufacturerModelName = "Osseo AI", "DXA-QC"
    ds.SeriesDescription = "AI quality control"
    ds.InstanceNumber, ds.SeriesNumber = 1, 9000
    ds.SpecificCharacterSet = "ISO_IR 192"
    return ds


def write_secondary_capture(bgr: np.ndarray, study_uid: str, source_uid: str, path: Path) -> None:
    import pydicom

    sop = derived_uid(source_uid, "sc")
    ds = _base_dataset("1.2.840.10008.5.1.4.1.1.7", sop, study_uid, derived_uid(study_uid, "sc-series"), "OT")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ds.ConversionType = "WSD"
    ds.Rows, ds.Columns = rgb.shape[:2]
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.BurnedInAnnotation = "YES"
    ds.PixelData = np.ascontiguousarray(rgb).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def write_sr(row: dict, study_uid: str, source_uid: str, source_sop_class: str, path: Path) -> None:
    """Basic Text SR with the verdict, violations and key measurements."""
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    def code(value, scheme, meaning):
        c = Dataset()
        c.CodeValue, c.CodingSchemeDesignator, c.CodeMeaning = value, scheme, meaning
        return Sequence([c])

    def text_item(name, value):
        it = Dataset()
        it.RelationshipType, it.ValueType = "CONTAINS", "TEXT"
        it.ConceptNameCodeSequence = code(name[0], "99OSSEO", name[1])
        it.TextValue = value[:1024]
        return it

    sop = derived_uid(source_uid, "sr")
    ds = _base_dataset("1.2.840.10008.5.1.4.1.1.88.11", sop, study_uid, derived_uid(study_uid, "sr-series"), "SR")
    ds.ValueType, ds.ContinuityOfContent = "CONTAINER", "SEPARATE"
    ds.ConceptNameCodeSequence = code("QC-REPORT", "99OSSEO", "DXA quality control report")
    ds.CompletionFlag, ds.VerificationFlag = "COMPLETE", "UNVERIFIED"
    ref = Dataset()
    ref.ReferencedSOPClassUID, ref.ReferencedSOPInstanceUID = source_sop_class, source_uid
    series = Dataset()
    series.SeriesInstanceUID = derived_uid(study_uid, "src-series")
    series.ReferencedSOPSequence = Sequence([ref])
    evidence = Dataset()
    evidence.StudyInstanceUID = study_uid
    evidence.ReferencedSeriesSequence = Sequence([series])
    ds.CurrentRequestedProcedureEvidenceSequence = Sequence([evidence])
    violations = [v for v in str(row.get("violation_codes", "")).split(";") if v]
    items = [
        text_item(("REGION", "Anatomical region"), str(row.get("anatomical_region", ""))),
        text_item(("VERDICT", "Quality verdict"),
                  "Нарушение качества" if row.get("quality_class") == 1 else "Качественное исследование"),
        text_item(("SCORE", "Violation probability"), f"{row.get('quality_prob', '')}"),
    ]
    for v in violations:
        items.append(text_item(("VIOLATION", "Detected violation"), f"{v}: {VIOLATION_RU.get(v, v)}"))
    items.append(text_item(("MEASUREMENTS", "Measurements"), str(row.get("measurements", ""))))
    items.append(text_item(("DISCLAIMER", "Disclaimer"),
                           "Результат работы ИИ-сервиса; требует подтверждения специалистом."))
    ds.ContentSequence = Sequence(items)
    ds.save_as(str(path), enforce_file_format=True)
