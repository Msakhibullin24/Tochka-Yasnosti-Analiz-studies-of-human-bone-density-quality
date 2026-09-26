"""Result table writers: CSV and XLSX (pure standard library, no optional dependencies)."""
from __future__ import annotations

import csv
import os
import tempfile
from functools import wraps
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

# The first eight columns are exactly the contract of the task statement (section 2.5), in order.
CONTRACT_COLUMNS = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
                    "violation_type", "processing_status", "time_of_processing"]
# Extra columns go after the contract so a strict positional/named parser is not disturbed.
# quality_prob: name and range [0;1] approved by the organiser (needed for ROC-AUC).
EXTRA_COLUMNS = ["quality_prob", "laterality", "violation_codes", "violation_scores", "violation_description", "measurements",
                 "region_confidence", "laterality_confidence", "pixel_mm", "pixel_mm_source", "duplicate_of", "error_code", "error_message", "path_to_file",
                 "explanation_png", "criterion_thresholds", "decision_reason", "decision_version", "pixel_mm_x", "row_id", "projection_assessment", "anatomy_assessment", "source_roi_assessment", "anatomical_checks_complete", "image_width", "image_height", "criterion_states", "violation_type_status", "review_reasons"]
EXTRA_COLUMNS.append('specialist_qc')
EXTRA_COLUMNS.append('specialist_outputs')
EXTRA_COLUMNS.append('requirement_checks')
EXTRA_COLUMNS.append('learned_anatomy')
EXTRA_COLUMNS.append('joint_evidence')


def columns(strict: bool) -> list[str]:
    """strict = the 8 contract columns + the organiser-approved quality_prob, nothing else."""
    return CONTRACT_COLUMNS + ["quality_prob"] if strict else CONTRACT_COLUMNS + EXTRA_COLUMNS


def atomic_output(writer):
    """Publish complete reports; a failed rewrite preserves the previous artifact."""
    @wraps(writer)
    def wrapped(rows, path, strict=False):
        path = Path(path)
        fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
        os.close(fd)
        try:
            writer(rows, Path(temporary), strict)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return wrapped


@atomic_output
def write_csv(rows: list[dict], path: Path, strict: bool = False) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns(strict), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _col(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


@atomic_output
def write_xlsx(rows: list[dict], path: Path, strict: bool = False) -> None:
    cols = columns(strict)
    numeric = {"quality_class", "time_of_processing", "quality_prob", "region_confidence",
               "laterality_confidence", "pixel_mm"}
    lines = []
    for r, row in enumerate([dict(zip(cols, cols))] + rows):
        cells = []
        for c, name in enumerate(cols):
            v = row.get(name, "")
            ref = f"{_col(c)}{r + 1}"
            if r > 0 and name in numeric and v not in ("", None):
                cells.append(f'<c r="{ref}"><v>{v}</v></c>')
            else:
                text = escape("" if v is None else str(v))
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>')
        lines.append(f'<row r="{r + 1}">{"".join(cells)}</row>')
    sheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
             + "".join(lines) + "</sheetData></worksheet>")
    parts = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>',
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>',
        "xl/workbook.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="results" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>',
        "xl/worksheets/sheet1.xml": sheet,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            # Fixed timestamp: the container bytes depend only on the rows (timing values still differ per run).
            z.writestr(zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0)), data)
