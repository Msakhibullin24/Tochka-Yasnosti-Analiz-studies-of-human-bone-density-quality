# Контракт сервиса анализа

Браузерный адаптер `src/services/analysis.ts` выполняет локальный технический pre-screening. Продукционная анатомическая модель подключается по этому контракту, сохраняя тип `Study` и provenance результата.

## `POST /api/v1/studies/analyze`

Загружает одно денситометрическое исследование в формате DICOM.

- Content-Type: `multipart/form-data`
- Поле файла: `file`
- Допустимые расширения: `.dcm`, `.dicom`
- Максимальный размер: 100 МБ
- Рекомендуемые ответы: `202` для асинхронной обработки либо `200` для синхронной

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

## `POST /api/v1/studies/compare`

Сравнивает текущее исследование с baseline после независимой оценки качества обоих исследований. Вход содержит `baselineStudyId`, `currentStudyId` и идентификатор записи LSC учреждения. BMD должен быть получен из проверенного структурированного источника или подтверждённого vendor-adapter, а не оцениваться по яркости preview.

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
- смена аппарата требует действующей записи cross-calibration, иначе возвращается `422 CROSS_CALIBRATION_REQUIRED`.

## Ошибки и наблюдаемость

- `400 INVALID_DICOM` — структура DICOM не читается;
- `413 FILE_TOO_LARGE` — превышен лимит;
- `415 UNSUPPORTED_TRANSFER_SYNTAX` — pixel stream нельзя декодировать;
- `422 UNSUPPORTED_PROTOCOL` — исследование не относится к поддерживаемым протоколам;
- `503 MODEL_UNAVAILABLE` — inference временно недоступен.

Каждый ответ сервера содержит `requestId`; логи хранят только псевдонимы, хеш модели, длительность этапов и коды ошибок. Исходные идентификаторы пациента, UIDs и Pixel Data в application-лог не попадают.

## Требования перед клиническим внедрением

1. Удалять прямые идентификаторы пациента до передачи в модель и вести журнал доступа без PHI.
2. Валидировать поддерживаемые DICOM-теги, фотометрическую интерпретацию и модели денситометров.
3. Версионировать модель, набор критериев и пороги; сохранять их в протоколе анализа.
4. Обеспечить ручное подтверждение результата врачом и аудит исправлений разметки.
5. Провести клиническую валидацию на независимой выборке и оформить ПО в соответствии с применимыми требованиями к медицинским изделиям.
