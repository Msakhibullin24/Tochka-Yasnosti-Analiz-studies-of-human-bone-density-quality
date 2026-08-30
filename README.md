# Osseo AI

Рабочий MVP сервиса автоматизированной оценки качества денситометрических изображений и анатомической разметки. Продукт принимает DICOM-исследование, показывает зоны анализа, классифицирует нарушения и формирует машиночитаемый протокол.

## Что реализовано

- адаптивное клиническое рабочее место со списком исследований;
- три состояния результата: качественно, нужна проверка, есть нарушения;
- интерактивный DXA-просмотрщик с анатомическими ROI, масштабом и отключением разметки;
- проверка позиционирования, анатомического охвата, разметки и артефактов;
- отдельный раздел анализа в динамике: baseline/current BMD, абсолютное и процентное изменение, LSC и quality gate сопоставимости;
- три безопасных сценария follow-up: значимое изменение, ручная проверка и запрет интерпретации;
- реальный разбор DICOM Part 10: метаданные, transfer syntax, геометрия и Pixel Data;
- локальный preview 8/16-битных несжатых MONOCHROME1/2 исследований без отправки файла;
- детерминированный технический pre-screening диапазона сигнала и целостности файла;
- Python/FastAPI ML-бэкенд для total-body DXA с интегрированной моделью `hawaii-ai/dxa-pointplacement`;
- автоматическая постановка 105 анатомических ориентиров, overlay и геометрические проверки центрирования, симметрии и охвата;
- безопасный protocol router: внешний checkpoint применяется только к total-body, а spine/hip остаются в режиме технической проверки до появления профильных моделей;
- проверяемый статус обезличивания, маскирование ID, UIDs, accession number и имени файла;
- provenance результата: режим, версия алгоритма, версия критериев и предупреждения;
- экспорт протокола в JSON;
- полная локализация RU/EN;
- клавиатурная навигация, видимый фокус, живые статусы и поддержка `prefers-reduced-motion`;
- тесты основных пользовательских сценариев.
- Dataset Workbench для обезличенных Hologic APEX P/R: manifest, processed/raw viewer, технический QC и patient-grouped splits;
- экспертная разметка дефектов по протоколу, L1–L4 landmarks, ROI и контуров с точным клавиатурным вводом координат;
- атомарное хранение нескольких экспертных чтений и выгрузка JSONL/COCO;
- явное исключение цветных Secondary Capture/печатных DICOM и отдельный PHI-безопасный журнал исключений;
- единый локальный запуск, полный verify-скрипт и Docker Compose.
- SHA-версия датасета и автоматический аудит отсутствующих/повреждённых ассетов,
  patient leakage, exact/source/near duplicates между split;
- измерение agreement нескольких экспертов и очередь adjudication, в которой
  конфликт закрывает только третье независимое чтение;
- tamper-evident PHI-free audit chain с request ID и автоматическая панель hard
  release gates для данных, моделей, RBAC и клинической валидации.

## Запуск

Требуется Node.js 22+.

```bash
npm install
npm run dev
```

Продукт откроется на `http://localhost:5173`.

Полный локальный контур frontend + API (API по умолчанию использует порт `8001`):

```bash
make dev
```

Подключение подготовленного датасета:

```bash
OSSEO_DATASET_ROOT=/secure/path/apex-dataset \
OSSEO_ANNOTATION_ROOT=/secure/path/apex-annotations \
make dev
```

Контейнерный запуск без ML-checkpoint:

```bash
mkdir -p data/dataset data/annotations data/longitudinal data/runtime
# Скопируйте содержимое обезличенного osseo-apex export в data/dataset
make docker-up
```

Интерфейс будет доступен на `http://localhost:8080`. Аннотации, longitudinal
registry и журналы хранятся в `data/annotations`, `data/longitudinal` и `data/runtime`; исходный dataset
подключается read-only. `make docker-up` передаёт UID/GID текущего пользователя,
чтобы сохранить приватные права файлов экспорта.

### ML-бэкенд total-body

Для базового API требуется Python 3.10–3.11. Пример с `uv`:

```bash
uv venv backend/.venv --python 3.11
backend/scripts/setup_ml.sh
backend/.venv/bin/python backend/scripts/fetch_dxa_checkpoint.py
cd backend && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

В режиме разработки Vite перенаправляет `/api` на `http://localhost:8000`.
Checkpoint занимает примерно 265 МБ, хранится в `backend/models/` и исключён из Git.
Исходные условия лицензирования checkpoint должны быть подтверждены перед его
перераспространением; детали находятся в `backend/THIRD_PARTY.md`.

Проверка перед публикацией:

```bash
npm run test
npm run build
npm run lint
backend/.venv/bin/python -m pytest -q backend/tests
# либо все проверки одной командой
make verify
```

## Архитектура

- `src/App.tsx` — рабочее место, просмотрщик, загрузка и экспорт;
- `src/data.ts` — локализация и демонстрационные исследования;
- `src/services/analysis.ts` — изолированный адаптер анализа;
- `src/services/dicom.ts` — безопасный DICOM parser и декодер несжатых grayscale-пикселей;
- `src/DatasetWorkbench.tsx` — браузер датасета, raw/processed viewer и разметка;
- `src/services/dataset.ts` — типизированный API-клиент dataset/annotation;
- `src/services/longitudinal.ts` — безопасные демонстрационные follow-up сценарии;
- `src/services/longitudinal-api.ts` и `src/DatasetLongitudinal.tsx` — рабочий registry/API/UI подтверждённых BMD, facility LSC и анализа в динамике;
- `src/services/ml-analysis.ts` — вызов ML API с безопасным fallback на локальный pre-screening;
- `backend/app/` — DICOM decode, MMPose inference, geometry QC и FastAPI;
- `backend/app/dataset.py` — manifest repository, annotation storage и COCO/JSONL export;
- `backend/app/evidence.py` — dataset integrity, inter-reader agreement, adjudication и release gates;
- `backend/app/audit.py` — PHI-free hash-chain операций и проверка целостности;
- `backend/app/longitudinal.py` — registry BMD/LSC/cross-calibration, baseline selection и deterministic comparison engine;
- `src/DatasetLongitudinal.tsx` — ввод подтверждённых BMD, временной ряд и безопасное сравнение;
- `backend/app/hologic_apex/` — P/R parser, de-identification, split и technical QC;
- `backend/third_party/dxa_pointplacement/` — зафиксированные исходники upstream под Apache 2.0;
- `src/types.ts` — единая доменная модель;
- `src/styles.css` — токены, адаптивность, состояния и доступность;
- `docs/API.md` — контракт для подключения ML-бэкенда.
- `docs/HOLOGIC_APEX_INGEST.md` — безопасный ingest проприетарных Hologic P/R, формат данных и CLI.
- `docs/WORLD_CLASS_METRICS.md` — полный протокол метрик, safety-gates, статистики и мониторинга;
- `docs/metrics.registry.json` — машиночитаемый реестр primary и hard-gate метрик.
- `docs/WINNING_ARCHITECTURE.md` — целевая multi-model архитектура, routing, data strategy и release gates.
- `docs/RUSSIA_PRODUCT_STRATEGY.md` — intended use, доказательная матрица, российская рамка и план на 90 дней.

## Граница готовности

Для total-body DXA подключена исследовательская модель 105 landmarks. Она обучена
на извлечённых air-ratio изображениях; применение к нормализованным DICOM требует
локальной валидации и всегда оставляет отдельный artifact review gate. Spine/hip
исследования проходят реальный технический pre-screening, но их позиционирование
и ROI по-прежнему переводятся на экспертную проверку до обучения профильных моделей.

### Hologic APEX P/R

Для пакетного аудита RAR или уже распакованной директории с парными файлами
`Pxx/Rxx` используется отдельный PHI-чувствительный контур:

```bash
backend/.venv/bin/osseo-apex /path/to/archive.rar
OSSEO_PSEUDONYM_KEY='secret-at-least-16-bytes' \
  make dataset INPUT=/path/to/archive.rar OUTPUT=/secure/path/apex-v1
```

Команда не изменяет исходник, не копирует P/R в результат, формирует keyed-HMAC
псевдонимы, PNG обработанных изображений и lossless `uint16` NPY формы
`height × logical_width × 6`. Физические названия шести transmission-каналов
намеренно не назначаются до эталонной верификации.

Экспорт дополнительно формирует `splits.json` и `qc_report.json`. Split строится
детерминированно по `patientGroupId`, поэтому исследования одного пациента не
попадают в разные части выборки.

План доведения до конкурсной и клинической модели и таксономия ошибок описаны в [docs/COMPETITION_STRATEGY.md](docs/COMPETITION_STRATEGY.md). Полная система primary/secondary/monitoring метрик, safety-gates и статистический протокол находятся в [docs/WORLD_CLASS_METRICS.md](docs/WORLD_CLASS_METRICS.md). Логика безопасного сравнения — в [docs/LONGITUDINAL_ANALYSIS.md](docs/LONGITUDINAL_ANALYSIS.md), полный продуктовый backlog — в [docs/PRODUCT_ROADMAP.md](docs/PRODUCT_ROADMAP.md), контракт inference-сервиса — в [docs/API.md](docs/API.md).
