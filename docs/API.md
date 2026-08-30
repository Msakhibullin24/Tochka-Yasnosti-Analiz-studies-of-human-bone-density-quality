# Контракт сервиса анализа

Браузерный адаптер `src/services/analysis.ts` выполняет локальный технический
pre-screening. `backend/app/main.py` реализует тот же контракт для total-body
landmark inference. Клиент `src/services/ml-analysis.ts` вызывает backend и
безопасно возвращается к локальному режиму при неподдерживаемом протоколе или
недоступной исследовательской модели.

## `POST /api/v1/studies/analyze`

Загружает одно денситометрическое исследование в формате DICOM.

- Content-Type: `multipart/form-data`
- Поле файла: `file`
- Допустимые расширения: `.dcm`, `.dicom`
- Максимальный размер: 100 МБ
- Рекомендуемые ответы: `202` для асинхронной обработки либо `200` для синхронной
- Необязательное multipart-поле `protocol_override`: `spine`, `hip`, `total-body`; ручной выбор сохраняется в routing trace

Пример результата:

```json
{
  "id": "ST-0248",
  "patientId": "P-80421",
  "filename": "DXA_LSP_0248.dcm",
  "acquiredAt": "2026-08-25T10:42:00+03:00",
  "type": "spine",
  "status": "passed",
  "score": 96,
  "confidence": 98,
  "technical": {
    "modality": "DX",
    "rows": 2800,
    "columns": 2200,
    "pixelSpacing": "0.20 × 0.20 mm",
    "photometricInterpretation": "MONOCHROME2",
    "transferSyntaxUid": "1.2.840.10008.1.2.1",
    "bitsAllocated": 16
  },
  "privacy": {
    "deidentificationVerified": true,
    "burnedInAnnotation": "NO"
  },
  "provenance": {
    "mode": "validated-model",
    "modelVersion": "osseo-mtl-1.0.0",
    "criteriaVersion": "DXA-QC-2026.1",
    "processedAt": "2026-08-25T07:42:05Z",
    "warnings": []
  },
  "criteria": [
    {
      "id": "position",
      "title": { "ru": "Позиционирование", "en": "Positioning" },
      "detail": {
        "ru": "Позвоночник расположен по средней линии.",
        "en": "Spine is centered."
      },
      "status": "pass",
      "confidence": 99
    }
  ],
  "recommendation": {
    "ru": "Повторное сканирование не требуется.",
    "en": "No repeat scan is required."
  }
}
```

## Правила статусов

- `passed`: все обязательные критерии пройдены;
- `review`: есть пограничные результаты, требуется экспертная проверка;
- `rejected`: найдено нарушение, влияющее на качество измерения.

Критерий использует `pass`, `warning` или `fail`. Цвет в интерфейсе не является единственным носителем статуса: каждому состоянию соответствуют текст и пиктограмма.

### Total-body extension

Для поддержанного total-body DICOM ответ содержит:

```json
{
  "type": "total-body",
  "routing": {
    "protocol": "total-body",
    "confidence": 0.55,
    "source": "dicom-rules",
    "evidence": ["body_part:WHOLE BODY"],
    "modelKey": "hawaii-ai/dxa-pointplacement@7ac19eb",
    "modelStatus": "ready"
  },
  "provenance": {
    "mode": "research-model",
    "modelVersion": "hawaii-ai-dxa-points-7ac19eb"
  },
  "landmarks": [
    { "name": "crown", "x": 0.5021, "y": 0.0184, "confidence": 0.992, "visible": true }
  ]
}
```

Координаты нормализованы в диапазон `[0, 1]` относительно исходного кадра.
Всего модель возвращает 105 ориентиров. Режим `research-model` нельзя заменять
на `validated-model` до локальной внешней валидации.

## `GET /api/v1/models`

Возвращает model registry и readiness каждого протокола. Статус `planned` не
разрешает inference и приводит к безопасному `422 UNSUPPORTED_PROTOCOL`.

## `POST /api/v1/studies/compare`

Сравнивает текущее исследование с baseline после независимой оценки качества обоих исследований. Вход содержит `baselineStudyId`, `currentStudyId` и идентификатор записи LSC учреждения. BMD должен быть получен из проверенного структурированного источника или подтверждённого vendor-adapter, а не оцениваться по яркости preview.

Рабочая реализация доступна как `POST /api/v1/longitudinal/compare`; старое имя
`/studies/compare` оставлено как первоначальный интеграционный контракт.

Перед сравнением registry заполняется через:

- `PUT /api/v1/longitudinal/measurements/{studyId}` — подтверждённые BMD и QC;
- `GET /api/v1/longitudinal/patients/{patientGroupId}/timeline` — временной ряд;
- `GET /api/v1/longitudinal/measurements/{studyId}/baseline-candidates` — предыдущие кандидаты;
- `PUT /api/v1/longitudinal/lsc-profiles/{profileId}` — versioned facility LSC;
- `PUT /api/v1/longitudinal/cross-calibrations/{calibrationId}` — межаппаратная калибровка.
- `GET /api/v1/longitudinal/comparisons/{comparisonId}` — сохранённый неизменяемый результат расчёта.

```json
{
  "baselineStudyId": "ST-0118",
  "currentStudyId": "ST-0248",
  "lscProfileId": "LSC-CLINIC12-HOLOGIC-2026",
  "status": "comparable",
  "confidence": 0.97,
  "assumed": false,
  "checks": [
    { "id": "protocol", "passed": true, "critical": true },
    { "id": "device", "passed": true, "critical": true },
    { "id": "cross-calibration", "passed": true, "critical": true },
    { "id": "positioning", "passed": true, "critical": false },
    { "id": "roi", "passed": true, "critical": false }
  ],
  "sites": [
    {
      "site": "l1-l4",
      "baselineBmd": 0.842,
      "currentBmd": 0.891,
      "absoluteChange": 0.049,
      "percentChange": 5.8,
      "lscPercent": 5.3,
      "status": "significant-gain"
    }
  ]
}
```

Правила отказа:

- `not-comparable`, если не пройден хотя бы один критический gate;
- `review`, если требуется коррекция укладки/ROI и сравнение можно пересчитать после подтверждения;
- `comparable` только после всех обязательных проверок;
- `significant-gain/loss` только при `comparable` и `|ΔBMD%| ≥ LSC%`;
- смена аппарата требует действующей записи cross-calibration, иначе результат имеет `status=not-comparable`, gate `cross_calibration=block` и не содержит интерпретируемого тренда.

## Ошибки и наблюдаемость

- `400 INVALID_DICOM` — структура DICOM не читается;
- `413 FILE_TOO_LARGE` — превышен лимит;
- `415 UNSUPPORTED_MEDIA_TYPE` — вход не распознан как DICOM-файл;
- `422 INVALID_PROTOCOL_OVERRIDE` — передан неизвестный ручной маршрут;
- `422 SECONDARY_CAPTURE_EXCLUDED` — цветной presentation/print DICOM исключён из обучающего контура и записан в PHI-безопасный реестр;
- `422 UNSUPPORTED_PROTOCOL` — исследование не относится к поддерживаемым протоколам;
- `503 MODEL_UNAVAILABLE` — inference временно недоступен.

Текущий application-код не журналирует исходные идентификаторы пациента, UIDs
или Pixel Data. В production ingress должен назначать `X-Request-ID`, а журналы —
хранить только этот идентификатор, псевдонимы, хеш модели, длительности этапов и
коды ошибок. Конфигурация reverse proxy требует отдельного privacy-аудита.

## Dataset Workbench API

Backend читает только экспорт `osseo-apex`, в котором
`privacy.directIdentifiersExported=false`. Исходные RAR/P/R и DICOM в этот
контур не подключаются.

- `GET /api/v1/datasets/current` — сводка, протоколы, split и число разметок;
- `GET /api/v1/datasets/current/integrity` — версия dataset, проверка ассетов, leakage и cross-split duplicates;
- `GET /api/v1/datasets/current/agreement` — покрытие, независимые чтения, Cohen κ и landmark disagreement;
- `GET /api/v1/datasets/current/adjudication` — очередь расхождений, требующих третьего эксперта;
- `GET /api/v1/datasets/current/studies` — фильтруемый manifest;
- `GET /api/v1/datasets/current/studies/{studyId}` — карточка и экспертные чтения;
- `GET /api/v1/datasets/current/studies/{studyId}/assets/{asset}.png` — processed raster;
- `GET /api/v1/datasets/current/studies/{studyId}/raw/{0..5}.png` — percentile preview raw-канала;
- `PUT /api/v1/datasets/current/studies/{studyId}/annotations` — валидированная разметка v1;
- `GET /api/v1/datasets/current/exports/coco` — геометрия COCO-style;
- `GET /api/v1/datasets/current/exports/annotations` — полные экспертные чтения JSONL;
- `GET /api/v1/exclusions` — PHI-безопасный реестр presentation-объектов.
- `GET /api/v1/readiness` — hard release gates версии `ru-dxa-qc/1.0.0`;
- `GET /api/v1/audit/status` — проверка hash-chain PHI-free audit trail без выдачи самих событий.

Каждый `/api`-ответ имеет `X-Request-ID`, `Cache-Control: no-store` и базовые
security headers. Сохранение разметки, выгрузка и исключение presentation-объекта
добавляют tamper-evident audit event. Файловая hash-chain обнаруживает изменение,
но не заменяет WORM-хранилище и журнал доступа production-контура.

Исходный `transmissions.npy` через HTTP не выдаётся. Preview raw нормализуется
только для визуального просмотра; значения в NPY не изменяются и сохраняют
контракт `height × logical_width × 6 uint16` с `phaseSemantics=unverified`.

## Требования перед клиническим внедрением

1. Удалять прямые идентификаторы пациента до передачи в модель и вести журнал доступа без PHI.
2. Валидировать поддерживаемые DICOM-теги, фотометрическую интерпретацию и модели денситометров.
3. Версионировать модель, набор критериев и пороги; сохранять их в протоколе анализа.
4. Обеспечить ручное подтверждение результата врачом и аудит исправлений разметки.
5. Провести клиническую валидацию на независимой выборке и оформить ПО в соответствии с применимыми требованиями к медицинским изделиям.
