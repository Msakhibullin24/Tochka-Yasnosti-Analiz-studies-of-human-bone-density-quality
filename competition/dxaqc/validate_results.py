"""Read-only acceptance checks: python -m dxaqc.validate_results --results results.csv."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .model import REGION_LABEL, VIOLATION_LABEL, CRITERIA
from .report import CONTRACT_COLUMNS


def validate(path, repeat=None, manifest=None, series=None, competition=False, timing=None):
    errors, warnings = [], []
    try:
        with open(path, encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError) as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": [], "images": 0}
    if competition and columns != CONTRACT_COLUMNS + ["quality_prob"]:
        errors.append("Competition columns must be exactly the eight contract fields plus quality_prob")
    missing = set(CONTRACT_COLUMNS) - set(columns)
    if missing:
        errors.append("Missing columns: " + ', '.join(sorted(missing)))
    if len(columns) != len(set(columns)):
        errors.append("Duplicate column names")
    if not rows:
        errors.append("Empty result table")
    times = defaultdict(float)
    labels = set(VIOLATION_LABEL.values())
    for i, r in enumerate(rows, 2):
        def fail(message):
            errors.append(f"Row {i}: {message}")
        status = r.get("processing_status")
        if None in r or any(v is None for v in r.values()):
            fail("malformed CSV row")
        if status not in ("Success", "Failure"):
            fail("invalid processing_status")
        try:
            duration = float(r.get("time_of_processing", ""))
            if not math.isfinite(duration) or duration < 0:
                raise ValueError()
            times[r.get("study_uid") or r.get("path_to_study") or f"row:{i}"] += duration
        except (ValueError, TypeError):
            fail("invalid processing time")
        if status == "Success":
            if r.get("anatomical_region") not in set(REGION_LABEL.values()):
                fail("unknown anatomical region")
            if r.get("quality_class") not in ("0", "1"):
                fail("invalid quality class")
            types = (r.get("violation_type") or "").split(';') if r.get("violation_type") else []
            if any(t not in labels for t in types) or len(types) != len(set(types)):
                fail("unknown or repeated violation type")
            group = 'spine' if r.get('anatomical_region') == REGION_LABEL['spine'] else 'hip'
            allowed = {VIOLATION_LABEL[k] for k in CRITERIA[group]}
            if not set(types) <= allowed:
                fail("violation type does not belong to anatomical region")
            if r.get('quality_class') == '0' and types:
                fail('quality class and violation types disagree')
            if r.get('quality_class') == '1' and not types:
                if competition:
                    fail('competition output requires a violation type for quality_class=1')
                elif 'violation_type_status' in columns and r.get('violation_type_status') != 'undetermined':
                    fail('missing violation type without explicit undetermined status')
                else:
                    warnings.append(f'Row {i}: violation detected but type undetermined; typification requirement is incomplete')
            if "quality_prob" in columns:
                try:
                    score = float(r.get("quality_prob", ""))
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError()
                except (ValueError, TypeError):
                    fail("invalid quality_prob")
        if status == "Failure" and "error_message" in columns and not r.get("error_message"):
            fail("failure without explanation")
        if status == 'Failure':
            if r.get('quality_class') not in ('', None):
                warnings.append(f'Row {i}: legacy Failure class is not a quality prediction')
            if r.get('violation_type'):
                fail('processing failure must not invent a violation type')
    for uid, seconds in times.items():
        if seconds > 180:
            errors.append(f"Study {uid}: processing time exceeds 180 seconds ({seconds:.3f})")
    if timing is not None:
        from .requirements import timing_report
        try:
            measured = json.loads(Path(timing).read_text())
            expected_timing = timing_report(rows, float(measured['batch_seconds']))
            if measured.get('version') != 1 or measured['study_image_seconds'].keys() != expected_timing['study_image_seconds'].keys():
                raise ValueError('timing study identities differ from table')
            for name in ('study_image_seconds', 'study_upper_bound_seconds'):
                for key, value in expected_timing[name].items():
                    actual = float(measured[name][key])
                    if not math.isfinite(actual) or abs(actual - value) > .001:
                        raise ValueError('timing differs from table or shared overhead')
            if not expected_timing['within_180_seconds']:
                errors.append('End-to-end study upper bound exceeds 180 seconds')
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(f'Invalid end-to-end timing: {exc}')
    else:
        warnings.append('End-to-end timing not checked: provide timing.json; table durations exclude shared overhead')
    if manifest is not None:
        if "path_to_file" not in columns:
            errors.append("path_to_file is required for input manifest comparison")
        elif Counter(r.get('path_to_file') for r in rows) != Counter(manifest):
            errors.append("Output paths do not match input manifest (including multiplicity)")
    else:
        warnings.append("Input completeness not checked: provide an independent path manifest")
    if "quality_prob" not in columns:
        warnings.append("quality_prob absent: ROC-AUC input not checked")
    if "error_message" not in columns:
        warnings.append("Failure explanations not available in this table")
    if repeat:
        with open(repeat, encoding="utf-8-sig", newline="") as stream:
            repeated = list(csv.DictReader(stream))
        fields = [c for c in ("path_to_file", "study_uid", "image_uid", "anatomical_region", "quality_class", "violation_type", "processing_status", "quality_prob", "measurements") if c in columns]
        original = Counter(tuple(r.get(k, '') for k in fields) for r in rows)
        if original != Counter(tuple(r.get(k, '') for k in fields) for r in repeated):
            errors.append("Repeated run changed predictions or image identities")
    if series:
        errors.extend(validate_series(series, rows))
    else:
        warnings.append("Additional DICOM series not checked")
    return {"valid": not errors, "profile": "competition_v2" if competition else "working_report", "errors": errors, "warnings": warnings, "images": len(rows),
            "success": sum(r.get('processing_status') == 'Success' for r in rows),
            "max_seconds_per_study": max(times.values(), default=0),
            "scope": "table contract, optional input manifest, repeat and DICOM source links; not clinical validation or full DICOM conformance"}


def validate_series(path, rows):
    import io
    import zipfile
    import pydicom
    errors, found = [], set()
    expected = {r['image_uid']: r.get('study_uid') for r in rows if r.get('processing_status') == 'Success' and r.get('image_uid')}
    if any(r.get('processing_status') == 'Success' and not r.get('image_uid') for r in rows):
        errors.append('Cannot check source links for successful rows without image_uid')
    if len({(r.get('image_uid'), r.get('study_uid')) for r in rows if r.get('processing_status') == 'Success' and r.get('image_uid')}) != len(expected):
        errors.append('An image_uid occurs in different studies')
    try:
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist()) > 2 * 1024**3:
                return ['Additional series exceed validation size limit (2 GiB)']
            for info in archive.infolist():
                if not info.filename.lower().endswith('.dcm'):
                    continue
                if info.file_size > 128 * 1024**2:
                    errors.append(f'{info.filename}: DICOM exceeds validation size limit')
                    continue
                ds = pydicom.dcmread(io.BytesIO(archive.read(info)), stop_before_pixels=True)
                refs = {str(e.value) for e in ds.iterall() if e.keyword == 'ReferencedSOPInstanceUID'}
                kind = 'sr' if str(ds.get('Modality', '')) == 'SR' else 'sc' if str(ds.get('SOPClassUID', '')) == '1.2.840.10008.5.1.4.1.1.7' else 'unknown'
                if not refs or not refs <= expected.keys() or kind == 'unknown':
                    errors.append(f'{info.filename}: missing/unknown source reference or unsupported derived object')
                for uid in refs & expected.keys():
                    if str(ds.get('StudyInstanceUID', '')) != expected[uid]:
                        errors.append(f'{info.filename}: source study does not match')
                    found.add((uid, kind))
    except Exception as exc:
        errors.append(f'Cannot validate additional series: {type(exc).__name__}: {exc}')
    missing = {(uid, kind) for uid in expected for kind in ('sc', 'sr')} - found
    if missing:
        errors.append(f'Missing SC/SR objects for {len(missing)} source/type pairs')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--repeat', type=Path)
    parser.add_argument('--manifest', type=Path, help='JSON array of expected relative input paths, independently collected')
    parser.add_argument('--series', type=Path, help='additional_series.zip: check readable SC/SR and source links')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timing', type=Path, help='timing.json from the same run, including shared preparation/publication')
    parser.add_argument('--competition', action='store_true', help='strict V2 submission contract; unresolved types are errors')
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text()) if args.manifest else None
        if manifest is not None and (not isinstance(manifest, list) or not all(isinstance(p, str) for p in manifest)):
            raise ValueError('manifest must be an array of paths')
        result = validate(args.results, args.repeat, manifest, args.series, competition=args.competition, timing=args.timing)
    except (OSError, ValueError, csv.Error) as exc:
        result = {"valid": False, "errors": [str(exc)]}
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding='utf-8')
    print(payload)
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
