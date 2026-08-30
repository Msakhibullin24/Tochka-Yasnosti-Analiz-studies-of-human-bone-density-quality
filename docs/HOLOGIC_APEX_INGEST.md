# Hologic APEX P/R ingest

## Назначение и граница доверия

`osseo-apex` — офлайн-контур подготовки проприетарных Hologic APEX-архивов.
Он намеренно отделён от публичного FastAPI inference endpoint: исходные P-файлы
и `index.mdb` могут содержать прямые идентификаторы, даты, демографию и сведения
об аппарате.

Поток данных:

```text
read-only RAR/directory
  → archive safety gate
  → exact P/R pairing
  → bounded TLV decoder
  → protocol + structural QC
  → keyed pseudonymization
  → atomic manifest + PNG + uint16 NPY
  → expert annotation / ML pipeline
```

Исходный архив не изменяется. В экспорт не копируются `Pxx`, `Rxx`,
`index.mdb`, `label.dat` и исходные имена файлов.

## Подтверждённая бинарная структура

P- и R-файлы состоят из little-endian TLV-записей:

```text
uint16 tag
uint32 total_length  # включает шестибайтовый заголовок
uint8 payload[total_length - 6]
```

Парсер отклоняет нулевые/чрезмерные размеры, выход записи за EOF, trailing bytes,
дубликаты singleton-тегов и массивы, размер которых не совпадает с заголовком.

### P-файл

- `0x0046` — обязательный обработанный 8-битный растр;
- `0x0165` — второй обработанный растр, присутствующий у spine/hip;
- payload изображения: `uint16 width`, `uint16 height`, `uint16 descriptor`,
  затем ровно `width × height` байт;
- `descriptor` не является `SamplesPerPixel`: значение `3` у hip по-прежнему
  соответствует одному байту на пиксель;
- `0x0150` содержит путь scan protocol, `0x0428/0x0429` — версию APEX.

Остальные известные и неизвестные записи сохраняются только в памяти ingest
процесса. Прямые идентификаторы не сериализуются в manifest.

### R-файл

- `0x003A` — ширина массива transmission samples;
- `0x003B` — число строк;
- `0x00CA` — little-endian `uint16` samples;
- каждая пространственная точка содержит шесть последовательных измерений.

Каноническая lossless-форма экспорта:

```text
dtype: uint16 little-endian
shape: [height, stored_width / 6, 6]
```

Порядок физических значений high/low × air/tissue/bone пока не доказан на
эталонном phantom/vendor-экспорте. Поэтому контракт использует индексы `0..5` и
`phaseSemantics: unverified`. Присваивать физические имена по визуальному сходству
запрещено.

## Обезличивание

Экспорт требует `OSSEO_PSEUDONYM_KEY` длиной минимум 16 UTF-8 байт. Идентификаторы
пациента, исследования и аппарата преобразуются независимо через
HMAC-SHA256 с namespace-разделением; в manifest сохраняются первые 160 бит.
Отпечатки исходных P/R также являются keyed HMAC, а не публичными SHA-256, чтобы
не создавать канал проверки присутствия конкретного файла в датасете.

Ключ:

- не передаётся аргументом командной строки;
- не записывается в dataset;
- должен храниться отдельно в secret manager;
- должен оставаться одинаковым для версий датасета, если требуется корректная
  patient-grouped дедупликация.

Audit без `--output` использует случайный временный ключ. Счётчики корректны,
но псевдонимы между запусками намеренно нестабильны.

## CLI

```bash
# Только аудит, без записи производных файлов
backend/.venv/bin/osseo-apex /path/to/new.rar

# Обезличенный spine dataset
OSSEO_PSEUDONYM_KEY='managed-secret-at-least-16-bytes' \
  backend/.venv/bin/osseo-apex /path/to/new.rar \
  --output /secure/datasets/apex-spine-v1 \
  --protocol spine_pa
```

Вместо RAR можно передать распакованную директорию. Если директория не содержит
P/R напрямую, но содержит ровно один RAR, CLI безопасно выберет его автоматически.
Печатные/Secondary Capture DICOM не подходят под P/R pairing и в dataset не входят.

Output directory должна отсутствовать. Команда отказывается перезаписывать
существующие данные и сначала собирает результат во временной директории.

```text
dataset_summary.json
manifest.jsonl
splits.json
qc_report.json
processed/ST-.../p_0046.png
processed/ST-.../p_0165.png
raw/ST-.../transmissions.npy
```

Каждая строка manifest соответствует
[`hologic-apex-manifest.schema.json`](hologic-apex-manifest.schema.json) и содержит
версию loader для воспроизводимости.

`splits.json` содержит детерминированное patient-grouped разбиение
train/validation/test. `qc_report.json` агрегирует только технические признаки:
динамический диапазон processed-растра, clipping/saturation, структурные флаги и
пригодность для последующей экспертной проверки. Baseline не оценивает BMD,
анатомическую укладку или корректность врачебного ROI.

## Профиль полученной выгрузки

На проверенной локальной выгрузке:

| Показатель | Значение |
|---|---:|
| P/R-пар | 26 |
| Уникальных patient/study-групп | 8 |
| Spine protocol | 10 |
| Hip | 9 |
| Forearm | 7 |
| Обработанных P-растров | 45 |
| Повреждённых TLV/raw payload | 0 |
| Коротких spine-серий для review | 1 |

Короткая spine-серия не удаляется автоматически: она сохраняется с
`spine_scan_unusually_short` и должна получить экспертную метку
`wrong_protocol`, `incomplete_coverage` или иной согласованный класс.

## ML-контракт

До подтверждения физики каналов допустимы два независимых baseline:

1. anatomy/QC baseline на обработанных `0x0046/0x0165`;
2. self-supervised/representation experiments на безымянном шестиканальном raw.

Для клинических BMD/ratio-карт требуется отдельный validation gate:

- установить точный порядок шести transmission measurements;
- проверить формулы на известном phantom;
- сопоставить результат с vendor BMD и ROI;
- задать допуски по каждой версии APEX и модели аппарата;
- сохранить derivation version и calibration provenance в manifest.

Текущие данные предназначены для parser validation и pilot annotation. Десяти
lumbar-сканов недостаточно для обучения или заявления
клинических метрик.
