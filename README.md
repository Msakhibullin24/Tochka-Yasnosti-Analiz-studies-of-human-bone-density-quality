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

## Запуск

Требуется Node.js 20+.

```bash
npm install
npm run dev
```

Продукт откроется на `http://localhost:5173`.

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
```

## Архитектура

- `src/App.tsx` — рабочее место, просмотрщик, загрузка и экспорт;
- `src/data.ts` — локализация и демонстрационные исследования;
- `src/services/analysis.ts` — изолированный адаптер анализа;
- `src/services/dicom.ts` — безопасный DICOM parser и декодер несжатых grayscale-пикселей;
- `src/services/longitudinal.ts` — расчёт ΔBMD, LSC-классификация и валидация временного ряда;
- `src/services/ml-analysis.ts` — вызов ML API с безопасным fallback на локальный pre-screening;
- `backend/app/` — DICOM decode, MMPose inference, geometry QC и FastAPI;
- `backend/third_party/dxa_pointplacement/` — зафиксированные исходники upstream под Apache 2.0;
- `src/types.ts` — единая доменная модель;
- `src/styles.css` — токены, адаптивность, состояния и доступность;
- `docs/API.md` — контракт для подключения ML-бэкенда.
- `docs/WORLD_CLASS_METRICS.md` — полный протокол метрик, safety-gates, статистики и мониторинга;
- `docs/metrics.registry.json` — машиночитаемый реестр primary и hard-gate метрик.
- `docs/WINNING_ARCHITECTURE.md` — целевая multi-model архитектура, routing, data strategy и release gates.

## Граница готовности

Для total-body DXA подключена исследовательская модель 105 landmarks. Она обучена
на извлечённых air-ratio изображениях; применение к нормализованным DICOM требует
локальной валидации и всегда оставляет отдельный artifact review gate. Spine/hip
исследования проходят реальный технический pre-screening, но их позиционирование
и ROI по-прежнему переводятся на экспертную проверку до обучения профильных моделей.

План доведения до конкурсной и клинической модели и таксономия ошибок описаны в [docs/COMPETITION_STRATEGY.md](docs/COMPETITION_STRATEGY.md). Полная система primary/secondary/monitoring метрик, safety-gates и статистический протокол находятся в [docs/WORLD_CLASS_METRICS.md](docs/WORLD_CLASS_METRICS.md). Логика безопасного сравнения — в [docs/LONGITUDINAL_ANALYSIS.md](docs/LONGITUDINAL_ANALYSIS.md), полный продуктовый backlog — в [docs/PRODUCT_ROADMAP.md](docs/PRODUCT_ROADMAP.md), контракт inference-сервиса — в [docs/API.md](docs/API.md).
