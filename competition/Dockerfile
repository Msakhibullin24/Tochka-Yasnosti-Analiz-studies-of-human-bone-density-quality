# Osseo AI DXA-QC: fully offline inference container (CPU; GPU is not required).
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=8 \
    DXAQC_DATA_DIR=/work \
    DXAQC_INPUT_ROOT=/data \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    JOBLIB_TEMP_FOLDER=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-lock.txt requirements-test.txt ./
# --no-deps: every package (direct and transitive) is pinned explicitly; the lock files are generated
# from the validated environment by walking package metadata, so the closure is complete.
RUN pip install --no-cache-dir --no-deps --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt -r requirements-lock.txt -r requirements-test.txt

COPY dxaqc ./dxaqc
COPY weights ./weights
COPY models ./models
COPY train.py ./train.py
COPY labels ./labels

# Integrity of shipped weights + import smoke test at build time (no network is used at run time).
RUN python -c "import hashlib,sys;from dxaqc.embedding import WEIGHTS,WEIGHTS_SHA256 as h;\
sys.exit(0 if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()==h else 'resnet18 weights checksum mismatch')" \
    && python -c "from dxaqc.pipeline import load_bundle; load_bundle(); import dxaqc.api"

RUN mkdir -p /work /data/in /data/out && chown -R 10001:10001 /work /data
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "dxaqc.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
