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
- interactive OpenAPI at `/docs`

## Environment

- `DXA_CHECKPOINT_PATH`
- `DXA_DEVICE=auto|cpu|cuda:0`
- `DXA_ALLOWED_ORIGINS=http://localhost:5173`
- `DXA_MAX_UPLOAD_BYTES`
- `DXA_MODEL_VERSION`
- `DXA_CRITERIA_VERSION`

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

GitHub Actions runs frontend and backend unit tests, linting, and the production
frontend build. It intentionally does not download the external 265 MiB research
checkpoint; full model smoke tests run locally after checksum verification.
