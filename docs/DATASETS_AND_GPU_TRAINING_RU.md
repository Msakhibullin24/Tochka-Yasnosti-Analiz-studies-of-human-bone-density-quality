# Данные и GPU для обучения Osseo AI

Актуальность проверки источников: **1 сентября 2026 года**.

Расширенный каталог до примерно 1,8 млн потенциальных 2D изображений и план
хранилища находятся в [LARGE_SCALE_DATASETS_RU.md](LARGE_SCALE_DATASETS_RU.md).

## Короткий ответ

Для целевой задачи Osseo AI нельзя просто объединить все найденные медицинские
изображения. Целевая модель проверяет качество **DXA позвоночника и проксимального
бедра**, поэтому данные делятся на четыре группы:

1. собственные взрослые DXA с экспертной разметкой — основной train/validation;
2. UK Biobank DXA — лучший внешний источник, но только после одобрения доступа;
3. MTDDH и BUU-LSPINE — вспомогательное предобучение геометрии, не клиническая
   валидация DXA;
4. Arak DXA — сначала карантин и аудит лицензии/персональных данных.

На текущем компьютере CUDA-обучение запустить нельзя: `nvidia-smi` не видит
NVIDIA GPU, в системе определяется только Intel Iris Xe. Свободно около 134 ГБ.
Здесь можно подготовить и проверить данные, но обучение следует запускать на
NVIDIA-машине с 12–24 ГБ VRAM и отдельным диском от 250 ГБ.

Важно: в репозитории сейчас есть inference для total-body, но ещё нет готового
CLI обучения spine/hip. Приведённые ниже действия готовят правильные данные и
окружение. После аудита первых файлов нужно реализовать dataloader, модели и
команды `train/evaluate` под фактический формат разметки.

## Что скачивать

| Приоритет | Набор | Что содержит | Как использовать | Ограничение |
| --- | --- | --- | --- | --- |
| P0 | Локальные DXA | взрослые spine/hip DICOM, vendor ROI, экспертные QC-метки | целевое обучение и локальная валидация | нужно организовать экспорт и разметку |
| P0 | [UK Biobank DXA](https://community.ukbiobank.ac.uk/hc/en-gb/articles/34219417027997-DXA-Regions-of-Interest-ROI) | raw DXA whole-body, hip, spine, LVA, knee; Field 20158 | внешнее предобучение/валидация после проверки доступных меток | доступ по заявке, MTA и оплате; ROI enCORE проприетарны |
| P0, аудит | [Ramathibodi BMD/VFA](https://data.mait.ai.in.th/dataset/bone-mineral-density-and-vertebral-fracture-asseessment-public-sharing) | lumbar/hip BMD, VFA и заявленные quality/positioning labels | потенциально самый близкий открытый target-QC набор | count/size не опубликованы, status ongoing; сначала проверить ZIP и деидентификацию |
| P1 | [MTDDH](https://www.kaggle.com/datasets/amirmmahdavikia/mtddh-pelvic-x-ray) | 1 250 тазовых рентгенограмм с 4 QC-регионами и 8 точками; ещё 906 с 8 точками | предобучение hip geometry, segmentation/keypoints, проверка пайплайна | детский plain X-ray, не adult DXA |
| P1 | [BUU-LSPINE](https://services.informatics.buu.ac.th/spine/) | 400 пар AP/lateral lumbar X-ray с координатами краёв позвонков | предобучение spine geometry и landmarks | plain X-ray, смешанный возраст, доступ к full по заявке |
| P2, карантин | [Arak Bone Densitometry](https://www.medrxiv.org/content/10.1101/2025.01.25.24319689v1) | около 4 020 DXA-изображений и профили 3 643 пациентов | изучить домен DXA; использовать только после аудита | метки диагноза, а не QC; лицензия данных и деидентификация требуют отдельного подтверждения |
| Уже в проекте | Hawaii AI total-body | checkpoint 105 landmarks, около 265 МБ | smoke-test существующего total-body маршрута | не применять к spine/hip |

### Что не скачивать сейчас

- `mahavisnuks/dexa-osteo` и `mahavisnuks/augment-dexa`: маленькая
  переупакованная/синтетическая выборка без достаточного происхождения и QC-меток;
- `orvile/knee-x-ray-osteoporosis-database`: колено, QUS/T-score, другая задача;
- `chinese-osteoporosis-dxa`: большой набор с недостаточной data card и признаками
  персональных данных в именах файлов — не переносить из карантина без правовой и
  privacy-проверки;
- hand masks, knee MRI, VerSe/CTSpine1K, RSNA lumbar MRI, MURA и Bone Age:
  другая модальность или анатомия, сейчас они только расходуют диск и увеличивают
  domain gap;
- Pseudo-DXA, YUHS-VERTE-X и DXA Data Xtraction Assistant не являются источниками
  готовых целевых весов для QC spine/hip;
- DepictQA не нужен для первого обучения: его natural-image IQA score нельзя
  считать клинической оценкой DXA;
- EvaluateSegmentation — инструмент офлайн-оценки масок, а не датасет и не модель.

## Минимальная конфигурация машины

Рекомендуемый стартовый узел:

- NVIDIA GPU с 16–24 ГБ VRAM;
- 32–64 ГБ RAM;
- NVMe от 500 ГБ, из них не менее 250 ГБ свободно под raw/derived/checkpoints;
- Linux, актуальный NVIDIA driver;
- Python 3.11 в отдельном окружении;
- PyTorch stable с CUDA, выбранной по официальному
  [инсталлятору PyTorch](https://pytorch.org/get-started/locally/).

Профили первого baseline:

| VRAM | Размер входа | Batch | Настройки |
| --- | ---: | ---: | --- |
| 8 ГБ | 512 px | 1 | AMP, gradient accumulation 16, ConvNeXt-Tiny/EfficientNetV2-S |
| 12 ГБ | 768 px | 1–2 | AMP, gradient checkpointing, accumulation 8 |
| 16 ГБ | 768–1024 px | 1–2 | AMP, accumulation 4–8 |
| 24 ГБ+ | 1024–1280 px | 2–4 | AMP, SegFormer-B2/ConvNeXt-S, accumulation по необходимости |

Это стартовые параметры, а не обещание качества. Финальный resolution и batch
выбираются по peak memory и learning curves. Для DXA нельзя без проверки уменьшать
изображение так, чтобы исчезали межпозвонковые границы или малые артефакты.

Проверка новой GPU-машины:

```bash
nvidia-smi
python3 --version
df -h
```

После установки PyTorch из официального selector:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
PY
```

Продолжать с GPU-обучением можно только при `cuda available: True`.

## Структура хранилища

Данные не должны лежать внутри Git-репозитория. На GPU-машине или внешнем диске:

```bash
export OSSEO_DATA_ROOT=/mnt/osseo-data
test -n "$OSSEO_DATA_ROOT"
mkdir -p "$OSSEO_DATA_ROOT"/{quarantine,raw/external,raw/local_dxa,derived,manifests,splits}
```

Рекомендуемая структура:

```text
/mnt/osseo-data/
├── quarantine/             # ещё не разрешённые к использованию источники
│   └── arak_dxa/
├── raw/
│   ├── external/
│   │   ├── mtddh/
│   │   └── buu_lspine/
│   └── local_dxa/
│       ├── spine/
│       ├── hip/
│       └── total_body/
├── derived/                # декодированные изображения/маски; всегда воспроизводимы
│   └── v1/
├── manifests/              # происхождение, лицензии, хэши, PHI-аудит
└── splits/                 # frozen patient-grouped folds
```

`raw/` считается immutable: нормализация, resize и конвертация пишутся только в
`derived/<version>`.

## Пошаговая загрузка

### 1. Инструменты загрузки

Создать отдельное окружение, не смешивая его с legacy inference-зависимостями:

```bash
uv venv .data-tools --python 3.11
uv pip install --python .data-tools/bin/python kaggle gdown
.data-tools/bin/kaggle auth login
```

Не добавлять токен Kaggle или `kaggle.json` в репозиторий.

### 2. MTDDH — скачать первым

Размер Kaggle-зеркала около 1,98 ГБ, лицензия на странице — CC BY 4.0.

```bash
.data-tools/bin/kaggle datasets files amirmmahdavikia/mtddh-pelvic-x-ray
.data-tools/bin/kaggle datasets download amirmmahdavikia/mtddh-pelvic-x-ray \
  --path "$OSSEO_DATA_ROOT/raw/external/mtddh" \
  --unzip
```

Зафиксировать источник и снимок файлов:

```bash
find "$OSSEO_DATA_ROOT/raw/external/mtddh" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OSSEO_DATA_ROOT/manifests/mtddh.sha256"
```

MTDDH разрешён только для auxiliary pretraining. Его изображения не входят в
adult DXA holdout и по ним нельзя заявлять клиническое качество Osseo AI.

### 3. BUU-LSPINE — запросить полный набор

1. Открыть [официальную страницу BUU](https://services.informatics.buu.ac.th/spine/).
2. Скачать sample и проверить изображения/CSV.
3. Заполнить Google Form для full `BUU-LSPINE(400)` с назначением:
   `non-commercial research on DXA acquisition quality, spine geometry pretraining`.
4. Сохранить письмо с разрешением и текст EULA в
   `manifests/licenses/buu_lspine/` вне публичного Git.
5. Распаковать в `$OSSEO_DATA_ROOT/raw/external/buu_lspine/` и посчитать SHA-256.

Не использовать диагнозы BUU как целевые DXA QC-классы. Полезны координаты краёв,
AP-геометрия и часть rotation/side-bending сигналов.

### 4. Arak DXA — сначала только инвентаризация в карантине

Статья сообщает около 4 020 PNG и клинические профили 3 643 пациентов. Лицензия
CC BY относится к препринту; перед обучением нужно подтвердить условия именно
файлов данных и проверить, нет ли PHI в изображении, таблицах и именах.

```bash
mkdir -p "$OSSEO_DATA_ROOT/quarantine/arak_dxa"
.data-tools/bin/gdown \
  'https://drive.google.com/drive/folders/1HmLTG4GFgB2s4D0x7TTRx8vV_VWY3sW3' \
  --folder \
  --json \
  > "$OSSEO_DATA_ROOT/quarantine/arak_dxa/inventory.json"
```

До скачивания всего набора просмотреть inventory и запросить у авторов:

- лицензию на сами изображения и таблицы;
- описание деидентификации;
- соответствие `patient ↔ image`, единицу наблюдения и наличие повторных визитов;
- формат/значение меток и протоколы spine/hip;
- разрешение на обучение и публикацию производных весов.

Только после письменного подтверждения и privacy-аудита:

```bash
.data-tools/bin/gdown \
  'https://drive.google.com/drive/folders/1HmLTG4GFgB2s4D0x7TTRx8vV_VWY3sW3' \
  --folder \
  --continue \
  -O "$OSSEO_DATA_ROOT/quarantine/arak_dxa/files/"
```

Даже после допуска Arak нельзя считать разметкой качества: osteoporosis,
osteopenia и normal — диагноз/состояние плотности, а не ошибки позиционирования,
ROI, движения или артефакты.

### 5. UK Biobank — начать заявку параллельно

UK Biobank предоставляет raw DXA через Field 20158; для проекта нужны как минимум:

- Field 20158 / Category 723 — raw DXA images;
- hip и lumbar-spine DXA;
- whole-body DXA для domain pretraining;
- доступные BMD/ROI derived fields из Categories 124/125;
- scanner/site/protocol metadata и repeat imaging, если разрешены проектом.

Порядок доступа: регистрация исследователя → application → collaborators →
одобрение → access fee и MTA → работа в UKB-RAP. Официальная
[инструкция по заявке](https://community.ukbiobank.ac.uk/hc/en-gb/articles/15453619166749-How-to-Complete-an-access-application).

Данные UK Biobank нельзя копировать в обычное локальное хранилище, если это не
разрешено MTA/RAP. Код обучения должен уметь запускаться рядом с данными в RAP.

### 6. Существующий total-body checkpoint

Это не обучение spine/hip, но обязательный smoke-test текущего маршрута:

```bash
uv venv backend/.venv --python 3.11
backend/scripts/setup_ml.sh
backend/.venv/bin/python backend/scripts/fetch_dxa_checkpoint.py
```

Скрипт проверяет размер и SHA-256 checkpoint. Условия распространения весов нужно
подтвердить до передачи третьим лицам.

## Какие собственные DXA нужны обязательно

Без локального целевого набора получится лишь демонстрация transfer learning.
Минимальная единица данных:

- исходный обезличенный DICOM, protocol `spine` или `hip`;
- стабильный псевдоним пациента, study/series UID после безопасной псевдонимизации;
- клиника, модель аппарата, версия ПО, дата и laterality;
- vendor ROI/контуры/точки, если экспортируются;
- `accept / review / repeat`;
- multi-label дефекты и severity;
- маска/polygon/keypoints для локализуемых ошибок;
- два независимых эксперта и adjudication третьим;
- причина `not-evaluable`.

Первый пилот: 100–200 исследований каждого маршрута для проверки схемы и
согласованности экспертов. Для baseline следует стремиться минимум к 500–1 000
исследований spine и 500–1 000 hip, расширяя выборку по learning curves и числу
редких дефектов. Это не порог клинической достаточности.

До обучения обязательны:

1. PHI-аудит DICOM tags, burned-in text, таблиц и имён файлов;
2. exact SHA-256 и perceptual-hash поиск дублей;
3. объединение всех визитов одного пациента в один fold;
4. стратификация по протоколу, дефектам, аппарату и клинике;
5. frozen external holdout, который никогда не используется для выбора модели;
6. проверка согласованности экспертов и утверждённый labelbook.

## Что именно обучать

Последовательность экспериментов:

1. **Smoke-test total-body** — проверить существующий checkpoint на нескольких
   обезличенных локальных DICOM, не дообучая его.
2. **Proxy pretraining** — MTDDH для hip segmentation/keypoints, BUU для spine
   edges/landmarks. Обучаются только anatomy heads/encoder.
3. **Target baseline** — отдельные `spine` и `hip` multi-task модели на локальных
   DXA: segmentation + keypoints + multi-label QC.
4. **Geometry features** — углы, coverage, ROI distance/IoU подаются в defect head.
5. **Patient-grouped 5-fold OOF** — пороги и calibration выбираются только по OOF.
6. **Frozen holdout/device shift** — финальный false-safe audit и срезы по аппарату.

Нельзя смешивать MTDDH/BUU с целевым holdout и нельзя делить изображения случайно
без группировки по пациенту.

## Когда данные готовы к реализации training CLI

Перед написанием dataloader достаточно передать разработчику:

- вывод `nvidia-smi` с GPU и VRAM;
- дерево каталогов первых 20–50 файлов каждого источника;
- по одному деидентифицированному примеру spine DICOM, hip DICOM и каждой
  annotation/ROI таблицы;
- описание полей Arak после аудита, если набор разрешён;
- подтверждение лицензий MTDDH, BUU, Arak/UKB;
- желаемый первый маршрут: рекомендуется начать со `hip`, потому что MTDDH даёт
  наиболее близкую публичную разметку геометрии.

После этого в проект нужно добавить отдельное training-окружение, валидатор
manifest/annotation schema, patient-grouped split generator, `train.py`,
`evaluate.py`, configs для VRAM-профилей и реестр экспериментов. До появления
этих компонентов команда «запустить обучение» в текущем репозитории отсутствует.
