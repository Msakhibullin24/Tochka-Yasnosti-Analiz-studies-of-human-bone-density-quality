# Osseo AI total-body DXA backend

FastAPI service integrating the Apache-2.0 source from
`hawaii-ai/dxa-pointplacement` at commit `7ac19eb9c99d9a5edc631446b45528e889d627ed`.

## Runtime flow

`DICOM decode → protocol router → grayscale normalization → MMPose → 105 landmarks → geometry QC → Study JSON`

Only total-body DXA is sent to the upstream checkpoint. Lumbar spine and proximal
femur require dedicated models and return `422 UNSUPPORTED_PROTOCOL`, which the
web client converts into an honest technical-screening fallback.

## Endpoints

- `GET /health`
- `GET /api/v1/models`
- `POST /api/v1/studies/analyze`, multipart field `file`
- `GET /api/v1/datasets/current` and `/studies`
- `GET /api/v1/datasets/current/integrity|agreement|adjudication`
- `GET /api/v1/datasets/current/studies/{studyId}` plus processed/raw image assets
- `PUT /api/v1/datasets/current/studies/{studyId}/annotations`
- `GET /api/v1/datasets/current/exports/coco|annotations`
- `GET /api/v1/exclusions`
- `GET /api/v1/readiness` and `/api/v1/audit/status`
- `PUT/GET /api/v1/longitudinal/measurements|lsc-profiles|cross-calibrations`
- `GET /api/v1/longitudinal/patients/{patientGroupId}/timeline`
- `POST /api/v1/longitudinal/compare`
- interactive OpenAPI at `/docs`

## Environment

- `DXA_CHECKPOINT_PATH`
- `DXA_DEVICE=auto|cpu|cuda:0`
- `DXA_ALLOWED_ORIGINS=http://localhost:5173`
- `DXA_MAX_UPLOAD_BYTES`
- `DXA_MODEL_VERSION`
- `DXA_CRITERIA_VERSION`
- `OSSEO_DATASET_ROOT` — a de-identified `osseo-apex` export; read-only at runtime
- `OSSEO_ANNOTATION_ROOT` — writable expert annotation directory
- `OSSEO_LONGITUDINAL_ROOT` — writable structured BMD, LSC and cross-calibration registry
- `OSSEO_EXCLUSION_LOG` — writable PHI-free Secondary Capture registry
- `OSSEO_AUDIT_LOG` — writable tamper-evident domain audit chain
- `OSSEO_AUDIT_HMAC_KEY` — separate secret, at least 16 bytes, for actor pseudonyms
- `OSSEO_AUTH_MODE=disabled|oidc|trusted-proxy` — release gate accepts only the latter two
- `OSSEO_CLINICAL_VALIDATION_ID` — accepted only as a stable `CVR-…` report ID

## Verification

```bash
scripts/setup_ml.sh
.venv/bin/python scripts/fetch_dxa_checkpoint.py
.venv/bin/python -m pytest -q tests
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The service never puts the source Patient ID, accession number, Study UID, or
Series UID into its JSON response. It does not log these fields in application
code. Reverse-proxy and infrastructure logging still require a separate privacy
review before clinical deployment.

Color presentation objects, including the four supplied RGB Secondary Capture
DICOM files, return `422 SECONDARY_CAPTURE_EXCLUDED`. The persistent registry
stores only a content-derived object ID, reason, size and timestamp—never the
source filename or DICOM identifiers.

## Hologic APEX batch ingest

The `osseo-apex` CLI is a separate offline trust boundary for proprietary Hologic
P/R archives. It validates RAR paths and unpacked size, pairs scans, parses the
little-endian TLV records, exports processed grayscale images and preserves the
raw six-transmission cube as lossless uint16 NPY.

Audit without writing derived data:

```bash
.venv/bin/osseo-apex /path/to/archive.rar
```

Export requires a secret from the environment so low-entropy patient identifiers
are not protected by an unsalted hash:

```bash
OSSEO_PSEUDONYM_KEY='secret-at-least-16-bytes' \
  .venv/bin/osseo-apex /path/to/archive.rar \
  --output /secure/path/dataset-v1 --protocol spine_pa
```

The output never includes original filenames, direct identifiers, `index.mdb`,
or source P/R files. See `docs/HOLOGIC_APEX_INGEST.md` for the data contract and
known scientific limitations.

GitHub Actions runs frontend and backend unit tests, linting, and the production
frontend build. It intentionally does not download the external 265 MiB research
checkpoint; full model smoke tests run locally after checksum verification.
