FROM node:22.14.0-bookworm-slim@sha256:1c18d9ab3af4585870b92e4dbc5cac5a0dc77dd13df1a5905cea89fc720eb05b AS ui
WORKDIR /ui
COPY package.json pnpm-lock.yaml ./
RUN npm install --global --no-fund --no-audit pnpm@12.5.1 \
    && pnpm install --frozen-lockfile
COPY index.html tsconfig*.json vite.config.ts ./
COPY src ./src
COPY public ./public
RUN pnpm run build

# Osseo AI DXA-QC: fully offline inference container (CPU; GPU is not required).
FROM ghcr.io/astral-sh/uv:0.12.3@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc AS uv
FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

COPY --from=uv /uv /uvx /bin/

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
COPY competition/requirements.txt competition/requirements-lock.txt competition/requirements-test.txt ./
# --no-deps: every package (direct and transitive) is pinned explicitly; the lock files are generated
# from the validated environment by walking package metadata, so the closure is complete.
RUN uv pip install --system --no-cache --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt -r requirements-lock.txt -r requirements-test.txt
RUN uv pip check --system
# Fail the build if repository resolution changes the reviewed system runtime.
RUN python -c "import subprocess; names=['gcc-12-base','libgcc-s1','libstdc++6','libgomp1']; versions=[subprocess.check_output(['dpkg-query','-W','-f=\${Version}',n],text=True) for n in names]; assert versions == ['12.2.0-14+deb12u1'] * 4, versions"

ENV DXAQC_WORKFLOW_PROFILE=/app/models/workflow_1_11_5/profile.json

COPY competition/dxaqc ./dxaqc
COPY docs/competition/specialist_sources.json ./specialist_sources.json
COPY competition/weights ./weights
COPY competition/models ./models
# Keep the offline image's research CLI surface in sync with the repository.
# The specialist tests and documented training commands import these modules
# from the application root; copying only train.py made a freshly built image
# fail before test collection completed.
COPY competition/*.py ./
COPY competition/experiments ./experiments
COPY competition/labels ./labels
COPY --from=ui /ui/dist ./dxaqc/static/app

# Integrity of shipped weights + import smoke test at build time (no network is used at run time).
RUN python -c "import hashlib,sys;from dxaqc.embedding import WEIGHTS,WEIGHTS_SHA256 as h;\
sys.exit(0 if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()==h else 'resnet18 weights checksum mismatch')" \
    && python -c "from dxaqc.pipeline import load_bundle; load_bundle(); import dxaqc.api"

RUN mkdir -p /work /data/in /data/out && chown -R 10001:10001 /work /data
USER 10001:10001
RUN python -c "from dxaqc.pipeline import Analyzer; a=Analyzer(); assert a.workflow_profile is not None"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/v1/ready', timeout=3)"
CMD ["uvicorn", "dxaqc.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
