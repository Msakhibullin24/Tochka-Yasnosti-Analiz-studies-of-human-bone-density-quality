# Расширенная архитектура Osseo AI — исследовательский резерв

> Обновление 17.09.2026: ниже сохранена расширенная архитектура, предложенная до аудита конкурсного набора. Текущий исполняемый план — [COMPETITION_STRATEGY.md](COMPETITION_STRATEGY.md). Сегментационные модели, patient-level validation и клиническая валидация здесь описывают будущие цели, а не имеющиеся результаты. Для текущих 252 уникальных изображений сначала нужны проверенные метки и компактный baseline. Геометрические аугментации из старого раздела 7 по умолчанию выключены: они могут менять целевые дефекты. JSON из старого demo flow не заменяет обязательный CSV/XLSX. Динамика BMD и total-body не входят в текущий конкурсный приоритет.

## 1. Стратегическое решение

Победная система строится не вокруг одной нейросети, а вокруг клинически
объяснимого каскада:

`DICOM safety gate → protocol/domain router → anatomy model → ROI comparison → artifact model → calibrated defect classifier → abstention → protocol + overlay`

Модель `hawaii-ai/dxa-pointplacement` занимает только маршрут `total-body`. Она
даёт 105 landmarks и геометрические признаки, но не должна применяться к
поясничному отделу или проксимальному бедру. Для этих протоколов нужны отдельные
multi-task модели, обученные на конкурсной экспертной разметке.

Главный продуктовый тезис: система не просто выдаёт score, а называет нарушение,
показывает его на изображении, объясняет измеримым признаком и умеет безопасно
отказаться от ответа.

## 2. Model registry и маршрутизация

| Маршрут | Модель | Выходы | Текущий статус |
| --- | --- | --- | --- |
| `total-body` | Hawaii AI ResNet-152/MMPose | 105 landmarks, confidence, geometry | интегрирована, research |
| `spine` | Osseo Spine MTL | L1–L4 masks, interspaces, axes, ROI, defects | требует данных/обучения |
| `hip` | Osseo Hip MTL | pelvis/femur masks, landmarks, axes, ROI, defects | требует данных/обучения |
| artifacts | общий detector/segmenter | металл, движение, внешние объекты, кальцинаты | требует данных/обучения |
| OOD | encoder ensemble + metadata checks | known/unknown domain, uncertainty | требует OOF features |

Каждый ответ сохраняет `routing.protocol`, `routing.confidence`, источник решения,
совпавшие DICOM-признаки, ключ модели и её readiness. Конфликтующие DICOM-теги не
разрешаются угадыванием: роутер возвращает `unsupported/review`.

Ручной override допустим только как аудируемое действие пользователя. Он не
увеличивает клиническую валидность модели и сохраняется как `manual-override`.

## 3. Spine pipeline

### Анатомические выходы

- instance segmentation L1, L2, L3, L4;
- маски межпозвонковых пространств;
- центральная линия позвоночника и ось кадра;
- граница кость/мягкие ткани;
- predicted reference ROI;
- visibility и uncertainty каждого позвонка.

### Вычисляемые признаки

- отклонение оси позвоночника от вертикали и центра;
- асимметрия педикул/поперечных отростков как признак ротации;
- полнота охвата L1–L4;
- пересечение ROI с межпозвонковым пространством;
- IoU и boundary distance между vendor ROI и reference ROI;
- подозрение на неверную нумерацию или исключение позвонка;
- локальные гипер-/гиподенсные области над костью.

### Классы нарушений

`spine_off_center`, `spine_tilt`, `spine_rotation`,
`spine_incomplete_coverage`, `vertebra_numbering`, `vertebra_roi`,
`vertebra_exclusion`, `edge_detection`, `hyperdense_artifact`,
`hypodense_artifact`, `motion_artifact`, `external_object`.

## 4. Hip pipeline

### Анатомические выходы

- segmentation головки, шейки, большого/малого вертела, диафиза и таза;
- ось диафиза и ось шейки;
- landmarks малого вертела и точек привязки ROI;
- predicted femoral-neck и total-hip ROI;
- soft-tissue boundary.

### Вычисляемые признаки

- угол оси диафиза;
- видимая площадь малого вертела как признак наружной ротации;
- перекрытие шейки седалищной костью;
- полнота охвата головки, шейки, вертелов и диафиза;
- положение нижней границы total-hip ROI;
- расстояние vendor ROI от reference landmarks;
- артефакты внутри кости и мягких тканей.

### Классы нарушений

`hip_rotation`, `hip_abduction_adduction`, `femur_axis`,
`hip_incomplete_coverage`, `femoral_neck_roi`, `total_hip_roi`,
`edge_detection`, `soft_tissue_boundary`, `hyperdense_artifact`,
`hypodense_artifact`, `motion_artifact`, `external_object`.

## 5. Использование Hawaii AI

Интегрированная модель выполняет четыре функции:

1. генерация реального total-body overlay;
2. центрирование и симметрия укладки по парным landmarks;
3. признаки сопоставимости повторных total-body исследований;
4. transfer-learning baseline для общей DXA-нормализации и landmark heads.

Нельзя напрямую переносить ResNet-152 weights на spine/hip и заявлять валидность.
Допустимый эксперимент — использовать backbone initialization и сравнить его с
ImageNet/self-supervised pretraining по полному patient-grouped OOF. Выбор
делается только по метрике и latency.

## 6. Данные, без которых задача не закрывается

Минимальная единица датасета:

- исходный DICOM и поддержанный transfer syntax;
- псевдоним пациента, клиника, аппарат, протокол и дата;
- исходная vendor-разметка/ROI, если она доступна;
- независимые чтения минимум двух экспертов;
- overall action: `accept/review/repeat`;
- multi-label типы дефектов и severity;
- masks/polygons/keypoints для локализуемых дефектов;
- причина `not-evaluable`;
- adjudication третьим экспертом.

До обучения выполняются UID/perceptual-hash leakage audit и patient-grouped split.
Все исследования одного пациента и near-duplicates обязаны находиться в одном fold.

## 7. Обучение

### Baseline

- отдельные spine и hip модели;
- encoder EfficientNetV2/ConvNeXt V2;
- U-Net/FPN/SegFormer decoder;
- heatmap keypoint head;
- multi-label classifier, получающий encoder features и geometry features;
- Dice+BCE для anatomy, Tversky для малых артефактов, asymmetric focal loss для дефектов;
- 5-fold stratified group CV.

### Улучшение

- hard-negative mining по false-safe случаям;
- active learning на disagreement экспертов и ансамбля;
- domain-aware normalization по аппарату без использования vendor как shortcut;
- class-specific thresholds;
- temperature scaling/isotonic calibration только по OOF;
- fold ensemble после ablation;
- OOD score по ensemble disagreement и embedding distance.

Аугментации ограничиваются небольшими rotation/translation, шумом, blur и
экспозицией. Flip, elastic deformation и crop, меняющий клинический label,
запрещаются.

## 8. Итоговое решение

- `passed`: протокол известен, все mandatory критерии ниже порогов, uncertainty мала;
- `review`: пограничная вероятность, OOD, конфликт роутера, низкая уверенность,
  неизвестная ROI или неподдержанный artifact head;
- `rejected`: критическое нарушение выше рабочего порога;
- тип нарушения сохраняется отдельным кодом с confidence, severity, action и
  геометрией локализации.

Score не является средней вероятностью. Это версионируемая функция клинических
стоимостей, где false-safe критического дефекта штрафуется сильнее false-positive.

## 9. Метрики первого места

| Уровень | Primary | Hard gate |
| --- | --- | --- |
| Итоговое решение | official metric + macro F1 | false-safe rate |
| Типы нарушений | per-class PR-AUC/F1 | sensitivity критических классов |
| Segmentation | Dice/surface Dice | HD95 |
| Landmarks | NME/PCK в мм | 95 percentile error |
| ROI correctness | IoU + boundary distance | sensitivity неправильной ROI |
| Calibration | ECE/Brier | risk-coverage |
| Router | macro F1 | критическая misroute rate |
| Domain shift | device/site slices | leave-one-device-out |
| Runtime | p50/p95 | failure rate и peak memory |

Метрики считаются с patient-level bootstrap 95% CI. Leaderboard не используется
как validation set.

## 10. Конкурсный demo flow

1. загрузка реального обезличенного DICOM;
2. показ DICOM safety и routing trace;
3. реальный anatomy/ROI overlay;
4. выбор конкретного нарушения и его локализации;
5. измеримое объяснение: угол, расстояние, coverage или ROI mismatch;
6. `passed/review/rejected` и действие оператору;
7. OOD/uncertainty и версия модели;
8. экспорт JSON;
9. пример безопасного отказа;
10. dashboard независимой валидации с CI и device slices.

## 11. Критический путь после получения датасета

1. schema/UID/duplicate audit и официальный evaluation harness;
2. статистика протоколов, классов, аппаратов и качества разметки;
3. frozen patient-grouped folds;
4. простые spine/hip baselines;
5. anatomy masks/keypoints;
6. geometry features и defect heads;
7. OOF calibration и пороги;
8. hard-negative mining;
9. frozen holdout/device-shift test;
10. packaging, overlays, error atlas и конкурсная репетиция.

### Release gate

Система не получает статус `validated-model`, пока отсутствуют независимый
holdout, patient-level CI, per-device slices, калибровка, false-safe audit и
подписанный экспертами labelbook.

## 12. Что уже реализовано

- DICOM Part 10 ingest и технический pre-screening;
- формальный роутер с abstention и manual override trace;
- model registry с readiness;
- total-body checkpoint и 105 landmarks;
- geometry QC для total-body;
- FastAPI и frontend adapter;
- overlay, JSON export, privacy/provenance;
- безопасный fallback spine/hip;
- longitudinal quality gates;
- backend/frontend tests и воспроизводимый CPU runtime.

Оставшийся главный разрыв — не инфраструктура, а экспертно размеченные spine/hip
данные и обучение профильных моделей.
