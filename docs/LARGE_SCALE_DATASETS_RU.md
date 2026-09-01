# Максимальный корпус данных для Osseo AI

Проверено по официальным страницам и первичным публикациям: **1 сентября 2026 года**.

## Вывод

2 ГБ MTDDH — действительно мало для большого предобучения. Это качественный
датасет для конкретной задачи pelvic-X-ray QC, но не весь корпус.

Реалистичный максимальный корпус можно довести до:

- **99 578 пакетов реальных DXA** UK Biobank для 94 678 участников;
- до восьми DXA-представлений на участника: whole-body bone/soft tissue, lumbar
  spine, lateral thoracolumbar spine, обе hip и обе knee проекции;
- около **128 тысяч** дополнительных костных X-ray/синтетических проекций с
  полезной разметкой;
- **874 414** больших общерадиологических изображений из четырёх независимых
  chest X-ray коллекций для self-supervised pretraining;
- более **3 500 3D CT volumes** с костными масками для генерации дополнительных
  DRR-проекций.

После получения UK Biobank и устранения пересечений это даёт ориентир до
**1,5–1,8 млн 2D изображений**, плюс синтетические DRR. Это верхняя оценка, а не
число независимых пациентов и не число изображений с целевыми QC-метками.

Главное ограничение остаётся прежним: открытых adult spine/hip DXA с экспертными
метками `accept/review/repeat`, позиционирования, ROI и артефактов почти нет.
Миллион чужих рентгенограмм не заменяет локальные целевые DXA.

## 1. Реальные DXA — максимальный приоритет

| Источник | Масштаб | Что реально полезно | Доступ | Решение |
| --- | ---: | --- | --- | --- |
| [UK Biobank Field 20158](https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=20158) | 99 578 bulk-пакетов, 94 678 участников, 5 153 repeat visits | whole-body, lumbar, LVA, обе hip/knee; BMD и множество derived fields | application, fee, MTA, UKB-RAP | **главный крупный источник** |
| [Ramathibodi BMD/VFA](https://data.mait.ai.in.th/dataset/bone-mineral-density-and-vertebral-fracture-asseessment-public-sharing) | count и archive size не опубликованы, status `ongoing` | lumbar/hip BMD, VFA severity и оценки резкости, позиционирования и technical quality | public ZIP, карточка указывает CC BY; регистрация может потребоваться | **P0 target-QC candidate; сначала inventory/audit** |
| [Arak Bone Densitometry](https://www.medrxiv.org/content/10.1101/2025.01.25.24319689v1) | около 4 020 PNG, 3 643 пациента | spine/hip/neck DXA и BMD/T-score | Google Drive, но лицензия файлов и PHI требуют подтверждения | карантин до аудита |
| [SOF](https://agingresearchbiobank.nia.nih.gov/studies/sof/details) | 10 366 женщин, многолетний cohort | DXA/BMD, переломы, spine/pelvis radiographs | NIA controlled access; raw DXA images на странице не гарантированы | запросить image inventory |
| [MrOS](https://agingresearchbiobank.nia.nih.gov/studies/mros/details) | 5 994 участника в US cohort, повторные визиты | hip/spine/whole-body DXA и fracture outcomes | NIA controlled access; raw images нужно подтвердить | запросить image inventory |
| [SWAN longitudinal BMD](https://agingresearchbiobank.nia.nih.gov/studies/swan/documents/download/Codebooks_and_Forms/nialbmd2019_cdbk.pdf/) | spine/hip до 16 follow-up visits | longitudinal BMD и QA flags | подтверждены таблицы; доступность raw images не подтверждена | полезно для метаданных/LSC |
| [MOST](https://agingresearchbiobank.nia.nih.gov/studies/most/details) | 4 551 участник, longitudinal | radiographs, MRI, BMD QA; часть hip data | controlled access | вторичный cohort, не основной DXA image source |
| [NHANES DXA](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DXX_J.htm) | национальные циклы 1999–2006 и 2011–2018 | BMD/body-composition tables и rigorous QC | таблицы открыты | для calibration/statistics, не для image training |

### Почему UK Biobank важнее всех остальных

Field 20158 содержит не только whole-body DXA, но также lumbar spine, LVA,
правую/левую hip и knee. Текущий публичный каталог не показывает суммарный размер
архива. До выгрузки нужно получить inventory средствами UKB-RAP; для распаковки,
derived PNG, масок и кэша следует предварительно резервировать **500 ГБ–1 ТБ**.

Ограничения:

- сканы преимущественно GE Lunar iDXA — это большой, но одновендорный домен;
- raw images не содержат готовых целевых QC-решений;
- ROI enCORE проприетарны и могут быть недоступны как удобные masks;
- UKB нельзя автоматически копировать на локальный диск вопреки MTA;
- оценка «до восьми изображений» не означает, что все восемь есть в каждом ZIP;
  точное число нужно получить инвентаризацией после выдачи доступа.

## 2. Наиболее близкие костные X-ray и segmentation datasets

| Набор | Изображения | Объём | Разметка | Роль в Osseo AI |
| --- | ---: | ---: | --- | --- |
| [PENGWIN Task 2](https://pengwin.grand-challenge.org/data/) | 50 000 в опубликованном training archive | около 38 ГБ в проверенном mirror | sacrum/hipbone/fragments masks, synthetic X-ray/DRR | **главный открытый hip segmentation pretrain** |
| [MURA](https://stanfordmlgroup.github.io/competitions/mura/) | 40 561 | проверить после RUA | normal/abnormal, 7 типов upper extremity | general bone X-ray encoder, OOD |
| [GRAZPEDWRI-DX](https://doi.org/10.6084/m9.figshare.14825193) | 20 327 | 15,2 ГБ | 67 771 boxes/polygons/lines: axis, metal, foreign body, text и др. | artifact/detector pretraining, но wrist/pediatric |
| [VinDr-SpineXR](https://physionet.org/content/vindr-spinexr/1.0.0/) | 10 466, 5 000 studies | DICOM; узнать после доступа | 13 abnormalities с radiologist boxes | **главный public spine X-ray detector pretrain** |
| [OAI](https://www.nia.nih.gov/research/resource/osteoarthritis-initiative-oai) | более 40 000 knee radiograph exams по protocol docs | большой controlled corpus | longitudinal OA readings | domain robustness и longitudinal ingest, не target anatomy |
| [MTDDH](https://www.nature.com/articles/s41597-025-05146-x) | 1 250 QC + 906 keypoint images | 1,98 ГБ в Kaggle mirror | 4 QC regions, 8 keypoints, 2 doctors + senior review | **самый близкий открытый QC-набор для hip** |
| [BUU-LSPINE](https://services.informatics.buu.ac.th/spine/) | подтверждено 400 AP/lateral pairs; на странице есть дополнительные releases 3 600/2 000 | по заявке | lumbar edges, diagnoses/position signals | spine landmarks и geometry |
| [FracAtlas](https://figshare.com/articles/dataset/The_dataset/22363012) | 4 083 | 323 МБ | fracture classification/boxes/masks; есть hip/hardware fields | artifact и localization smoke-test |
| [CGMH Pelvis Segmentation](https://github.com/yaufan/Pelvis-X-ray_Segmentation_Database) | 400 | небольшой | hip masks на AP pelvis | real pelvis segmentation validation |
| [Pelvic tilt benchmark](https://www.nature.com/articles/s41597-024-04003-7) | 115 | небольшой | 5 annotators, landmark uncertainty | проверка допустимой ошибки landmarks |

Сумма таблицы без OAI составляет примерно 128 тысяч изображений/проекций. Она
намного ближе к Osseo AI, чем chest X-ray, хотя часть данных синтетическая или с
другой анатомией.

### Конфликт версии PENGWIN

Официальная challenge-страница ведёт на Zenodo Task 2: 100 CT × 500 проекций =
50 тысяч image/mask pairs. Перед загрузкой нужно сохранить Zenodo record JSON,
version, MD5 и фактическое число файлов. Нельзя смешивать PENGWIN Task 1/2 как
независимых пациентов: X-ray синтезированы из CT тех же cases. Метаданные Zenodo
указывают CC BY 4.0, а summary paper — более строгую CC BY-NC-SA; до ответа авторов
следует соблюдать более строгие условия и не использовать набор коммерчески.

## 3. Большой общерадиологический pretraining

Эти наборы дают много реальных рентгенограмм и разнообразие аппаратов, экспозиции,
кропов и позиционирования. Но это преимущественно грудная клетка. Их допустимая
роль — self-supervised encoder pretraining, DICOM robustness и обучение на
искусственных distortions. Их disease labels не переносятся в DXA heads.

| Набор | Масштаб | Ориентировочный объём | Условия |
| --- | ---: | ---: | --- |
| [MIMIC-CXR-JPG v2.1](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) | 377 110 images, 227 827 reports | сотни ГБ; DICOM ещё больше | PhysioNet credentialing + DUA |
| [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | 224 316 images, 65 240 patients | около 450 ГБ full-resolution; small release около 11 ГБ | Stanford RUA |
| [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/) | 160 868+ images, 67 000 patients | 1,02 ТБ full dataset | research-use application |
| [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) | 112 120 images, 30 805 patients | десятки ГБ | public/CC0 mirror, проверить original README |

Итого: **874 414** изображений. Это четыре разных клинических источника, но не
четыре DXA-датасета.

### Что выбрать вместо скачивания всех 2 ТБ chest X-ray

Первый ablation следует провести на готовых radiography weights из
[TorchXRayVision](https://github.com/mlmed/torchxrayvision) и на небольшом
patient-grouped подмножестве MIMIC. Если они не улучшают OOF на target DXA,
скачивание PadChest/CheXpert full не оправдано.

Можно также сравнить:

- [RadImageNet weights](https://github.com/BMEII-AI/RadImageNet), обученные на
  1,35 млн CT/MRI/US images; это не X-ray и только экспериментальная инициализация;
- [BiomedCLIP](https://www.microsoft.com/en-us/research/publication/biomedclip-a-multimodal-biomedical-foundation-model-pretrained-from-fifteen-million-scientific-image-text-pairs/),
  обученный на 15 млн scientific image-text pairs; использовать веса, а не
  скачивать PMC-15M, и обязательно сравнивать с ImageNet/DINO/Rad-Xray baselines.

## 4. 3D datasets для генерации размеченных DRR

| Набор | Масштаб | Разметка | Что генерировать |
| --- | ---: | --- | --- |
| [CTPelvic1K](https://github.com/MIRACLE-Center/CTPelvic1K) | 1 184 CT volumes, более 320 тыс. slices | lumbar spine, sacrum, left/right hip | AP pelvis/hip DRR, rotation, crop, metal-artifact variants |
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | 1 204 CT subjects | 104 structures, включая L1–L5, sacrum, hips и femurs | lumbar/hip DRR с точными projected masks |
| [CTSpine1K](https://arxiv.org/abs/2105.14711) | 1 005 CT volumes, более 11 100 vertebrae | vertebrae masks | AP/lateral lumbar DRR и landmarks |
| [PENGWIN Task 1](https://pengwin.grand-challenge.org/data/) | 150 pelvic CT patients | pelvic fracture fragment masks | источник уже опубликованных Task 2 DRR |

Из 3 000 независимых CT можно сгенерировать сотни тысяч проекций с идеальными
масками и известными углами. Но они остаются synthetic data: их нельзя использовать
как независимый clinical holdout, а overlapping source cohorts нужно дедуплицировать.

Рекомендуемые управляемые вариации DRR:

- AP, небольшие rotation/tilt и abduction/adduction;
- регулируемый incomplete coverage без изменения anatomy identity;
- metal/hardware и внешние объекты отдельными слоями;
- blur/noise/exposure/compression с сохранёнными параметрами;
- projected L1–L4, sacrum, femur/head/neck/trochanter masks и landmarks.

Не следует имитировать клиническую достоверность только визуально: распределение
DXA-сигнала и scatter должно калиброваться по реальным target DXA.

## 5. Наборы с метками качества изображений

Открытый большой корпус именно с клинической оценкой качества встречается редко:

- MTDDH: 1 250 pelvic QC images — наиболее релевантный public ground truth;
- [дополнение к MIMIC-CXR](https://plos.figshare.com/articles/dataset/Updated_meta_data_file_for_MIMIC-CXR_dataset_377_110_images_/29112355)
  содержит metadata для 377 110 изображений и поле
  `image quality`, но ручные QA labels относятся только к выделенным evaluation
  subsets; нельзя считать все 377 тысяч экспертно размеченными;
- GRAZPEDWRI-DX размечает `axis`, `metal`, `foreignbody`, `text` и soft-tissue
  объекты — полезные surrogate tasks;
- VinDr-SpineXR содержит `surgical implant` и `foreign body` boxes;
- PENGWIN позволяет точно создавать labels геометрических нарушений, потому что
  camera pose известна.

Для большого QC-корпуса правильнее взять реальные radiographs без QC-labels и
создать **параметрические пары** `clean → corrupted`, сохранив severity и mask.
Затем fine-tune только на локальных expert DXA. Natural-image IQA labels или
DepictQA score не являются клиническим ground truth.

## 6. План хранилища

На текущем компьютере свободно около 134 ГБ — этого недостаточно даже для
безопасной распаковки первого большого пакета.

| Профиль | Что входит | Raw | Рабочий диск с derived/cache/checkpoints |
| --- | --- | ---: | ---: |
| Target-first | MTDDH, PENGWIN, VinDr-SpineXR, GRAZ, FracAtlas, CGMH, BUU | примерно 80–150 ГБ | 500 ГБ минимум |
| Research | target-first + NIH + MIMIC subset + CTPelvic/CTSpine | 0,5–1,5 ТБ | 2–4 ТБ |
| Maximum local | MIMIC full + CheXpert full + PadChest + NIH + все open MSK/CT | 2–3 ТБ raw | **4 ТБ минимум, 8 ТБ комфортно** |
| UKB | Field 20158 и derived assets внутри RAP | точный объём проверить средствами RAP после одобрения | 0,5–1 ТБ project storage резервом |

Необходимо учитывать двойной объём при распаковке архивов и ещё 0,5–1× raw для
derived 16-bit PNG/WebDataset shards. Один dataset нельзя одновременно хранить в
DICOM, JPG, PNG и NumPy без зафиксированной причины.

## 7. Рекомендуемая очередь получения

### Сейчас: заявки и открытые релевантные данные

1. Подать UK Biobank application на Field 20158 и связанные DXA fields.
2. Зарегистрироваться в Thailand Medical AI Data Sharing Platform, получить
   inventory Ramathibodi `osteoporosis.zip`, проверить image count, schema,
   деидентификацию и соответствие заявленных QC-полей фактическим файлам.
3. Получить PhysioNet credentialing и подписать DUA для VinDr-SpineXR/MIMIC-CXR.
4. Скачать PENGWIN Task 2, MTDDH, GRAZPEDWRI-DX, FracAtlas и CGMH Pelvis.
5. Запросить BUU-LSPINE full.
6. Направить NIA запросы в SOF/MrOS/MOST с отдельным вопросом о доступности raw
   DXA image archives, scan-analysis files и QA flags.
7. Провести Arak license/PHI audit до полного скачивания.

### После добавления диска и первого baseline

8. Скачать NIH ChestX-ray14 и patient-grouped subset MIMIC-CXR-JPG.
9. Скачать/собрать CTPelvic1K и CTSpine1K, создать контролируемый DRR generator.
10. Сравнить ImageNet, TorchXRayVision, RadImageNet и self-supervised target
   pretraining на одинаковых frozen folds.
11. Только если большой CXR pretraining улучшает target OOF, загружать CheXpert и
    PadChest full.

## 8. Download manifest — обязательные поля

До скачивания каждого архива создать запись:

```text
dataset_id, canonical_url, record_version, retrieved_at, license,
dua_id, modality, anatomy, image_count_claimed, patient_count_claimed,
archive_size, sha256, access_scope, commercial_use, redistribution,
phi_audit_status, lineage_parent, allowed_role
```

`allowed_role` принимает только:

- `target_train` — реальные expert-labelled DXA;
- `target_holdout` — отдельная клиника/аппарат, никогда не участвует в выборе;
- `proxy_pretrain` — X-ray/DRR/CT;
- `ood` — другая анатомия или модальность;
- `quarantine` — нет подтверждённой лицензии/privacy;
- `statistics_only` — только таблицы BMD/популяционные показатели.

## 9. Практическое решение

Для Osseo AI не нужно обучать одну модель на всех 1,8 млн изображений. Нужен
каскад:

1. radiography SSL encoder на большом разрешённом корпусе;
2. anatomy pretraining на PENGWIN/VinDr/MTDDH/BUU и DRR;
3. отдельный target fine-tuning spine/hip на реальных DXA;
4. отдельный artifact/QC head на expert labels;
5. финальная оценка только на frozen real-DXA holdout по пациентам и аппаратам.

Это даёт пользу от огромного объёма и не подменяет целевую медицинскую задачу
нерелевантными labels.
