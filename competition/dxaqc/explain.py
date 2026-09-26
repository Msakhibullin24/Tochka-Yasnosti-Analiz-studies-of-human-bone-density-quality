"""Visual and textual explanations: overlay image, DICOM Secondary Capture series, DICOM SR.

Generated instance UIDs bind the source SOPInstanceUID and report content. Repeated
content has the same identity; a changed decision or overlay gets a new identity. No patient attributes are copied except the (already anonymised)
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
    return int(round((x + .5) * scale[0] - .5)), int(round((y + .5) * scale[1] - .5))


def render_overlay(pixels: np.ndarray, region: str, overlay: dict, features: dict, violations: list[str],
                   quality_class: int, score: float, pixel_mm: float, pixel_mm_x: float | None = None, row: dict | None = None) -> np.ndarray:
    """Render in physical proportions onto square display pixels, including the banner."""
    h, w = pixels.shape
    sx, sy = pixel_mm_x or pixel_mm, pixel_mm
    if not all(np.isfinite(v) and v > 0 for v in (sx, sy)):
        raise ValueError("invalid display calibration")
    density = min(2.0 / min(sx, sy), 2200 / max(w * sx, h * sy))
    size = (max(1, round(w * sx * density)), max(1, round(h * sy * density)))
    scale = (size[0] / w, size[1] / h)
    canvas = cv2.cvtColor(cv2.resize(pixels, size, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
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
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
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
        m3, m2 = int(30 / pixel_mm * scale[1]), int(20 / (pixel_mm_x or pixel_mm) * scale[0])
        H, W = canvas.shape[:2]
        cv2.line(canvas, (0, m3), (W, m3), (160, 160, 160), 1)
        cv2.line(canvas, (0, H - m3), (W, H - m3), (160, 160, 160), 1)
        xl = m2 if region == "hip_right" else W - m2
        cv2.line(canvas, (xl, 0), (xl, H), (160, 160, 160), 1)
        lines.append(f"shaft angle {features.get('shaft_angle_deg', float('nan')):+.1f} deg")
        lines.append(f"lesser trochanter {features.get('lesser_troch_protrusion_mm', float('nan')):.1f} mm")
        lines.append(f"model features: top {features.get('troch_top_margin_mm', float('nan')):.0f} / bottom "
                     f"{features.get('ischium_bottom_margin_mm', float('nan')):.0f} / lateral "
                     f"{features.get('lateral_margin_mm', float('nan')):.0f} mm")
    verdict = f"MODEL: {'FLAGGED' if quality_class else 'NOT FLAGGED'} score={score:.2f}"
    lines.insert(0, 'ANATOMICAL CHECKS INCOMPLETE')
    if quality_class and not violations:
        lines.insert(1, 'Violation type UNDETERMINED')
    if row:
        from .result_context import parsed
        projection = parsed(row, 'projection_assessment', {})
        source = parsed(row, 'source_roi_assessment', {})
        if region == 'spine':
            lines.append('Th12: NOT LOCALIZED')
        lines.extend(['Projection: ' + projection.get('status', 'unavailable'),
                      'Source ROI: ' + source.get('status', 'unavailable'),
                      'Landmarks are unvalidated candidates'])
    canvas = cv2.copyMakeBorder(canvas, 0, 0, 0, max(0, 640-canvas.shape[1]), cv2.BORDER_CONSTANT, value=(0,0,0))
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


def write_secondary_capture(bgr: np.ndarray, study_uid: str, source_uid: str, path: Path,
                            source_sop_class: str = "1.2.840.10008.5.1.4.1.1.1") -> None:
    import pydicom

    sop = derived_uid(source_uid, "sc:" + hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest())
    ds = _base_dataset("1.2.840.10008.5.1.4.1.1.7", sop, study_uid, derived_uid(study_uid, "sc-series"), "OT")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ds.ConversionType = "WSD"
    ds.PixelAspectRatio = [1, 1]  # render_overlay already resampled anatomy to physical proportions
    ds.Rows, ds.Columns = rgb.shape[:2]
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.BurnedInAnnotation = "YES"
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    ref = Dataset()
    ref.ReferencedSOPClassUID, ref.ReferencedSOPInstanceUID = source_sop_class, source_uid
    ds.SourceImageSequence = Sequence([ref])
    ds.PixelData = np.ascontiguousarray(rgb).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def write_sr(row: dict, study_uid: str, source_uid: str, source_sop_class: str, path: Path,
             source_series_uid: str = "") -> None:
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

    import json
    stable_keys = ('anatomical_region', 'quality_class', 'quality_prob', 'violation_codes', 'violation_type',
                   'measurements', 'decision_version', 'projection_assessment', 'anatomy_assessment',
                   'source_roi_assessment', 'criterion_states', 'review_reasons', 'requirement_checks')
    content_hash = hashlib.sha256(json.dumps({k: row.get(k) for k in stable_keys}, sort_keys=True,
                                           ensure_ascii=False).encode()).hexdigest()
    sop = derived_uid(source_uid, "sr:v3:" + content_hash)
    ds = _base_dataset("1.2.840.10008.5.1.4.1.1.88.11", sop, study_uid, derived_uid(study_uid, "sr-series"), "SR")
    ds.ValueType, ds.ContinuityOfContent = "CONTAINER", "SEPARATE"
    ds.ConceptNameCodeSequence = code("QC-REPORT", "99OSSEO", "DXA quality control report")
    ds.CompletionFlag, ds.VerificationFlag = "COMPLETE", "UNVERIFIED"
    ref = Dataset()
    ref.ReferencedSOPClassUID, ref.ReferencedSOPInstanceUID = source_sop_class, source_uid
    series = Dataset()
    ds.SourceImageSequence = Sequence([ref])
    series.SeriesInstanceUID = source_series_uid
    series.ReferencedSOPSequence = Sequence([ref])
    evidence = Dataset()
    evidence.StudyInstanceUID = study_uid
    evidence.ReferencedSeriesSequence = Sequence([series])
    if source_series_uid:
        ds.CurrentRequestedProcedureEvidenceSequence = Sequence([evidence])
    from .result_context import model_verdict, assessment_notes
    violations = [v for v in str(row.get("violation_codes", "")).split(";") if v]
    items = [
        text_item(("REGION", "Anatomical region"), str(row.get("anatomical_region", ""))),
        text_item(("VERDICT", "Quality verdict"),
                  model_verdict(row)),
        text_item(("SCORE", "Model score"), f"{row.get('quality_prob', '')}"),
    ]
    for v in violations:
        items.append(text_item(("VIOLATION", "Detected violation"), f"{v}: {VIOLATION_RU.get(v, v)}"))
    items.append(text_item(("MEASUREMENTS", "Measurements"), str(row.get("measurements", ""))))
    items.append(text_item(("DISCLAIMER", "Disclaimer"),
                           "Результат работы ИИ-сервиса; требует подтверждения специалистом."))
    items.append(text_item(("QC-LIMITS", "Assessment limits"), "\n".join(assessment_notes(row))))
    from .result_context import parsed
    for check in parsed(row, 'requirement_checks', []):
        items.append(text_item(('REQ-CHECK', 'Required check'),
                               f"{check['id']}: {check['status']}; {check.get('reason', '')}"))
    for key in ('projection_assessment', 'anatomy_assessment', 'source_roi_assessment', 'criterion_states', 'review_reasons'):
        if row.get(key):
            items.append(text_item((key.upper().replace('_', '-')[:16], key), str(row[key])))
    ds.ContentSequence = Sequence(items)
    ds.save_as(str(path), enforce_file_format=True)
