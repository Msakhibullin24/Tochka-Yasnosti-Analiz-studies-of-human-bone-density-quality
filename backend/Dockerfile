# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
COPY third_party ./third_party
RUN --mount=type=cache,target=/root/.cache/pip pip install .

RUN mkdir -p /data/annotations /data/longitudinal /data/runtime && chown -R 10001:10001 /data
USER 10001:10001

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
