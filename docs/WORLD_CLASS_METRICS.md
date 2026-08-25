# Система метрик Osseo AI мирового уровня

## 1. Назначение

Этот документ задаёт единый протокол оценки Osseo AI: от качества данных и экспертной разметки до ML, longitudinal-анализа, клинической пользы, DICOM-интеграции, безопасности и эксплуатации. Он не содержит заявлений о достигнутой точности: численные результаты можно публиковать только после заморозки данных, модели и порогов.

Главный принцип: официальная метрика конкурса определяет место в рейтинге, но не имеет права отменять safety-gates. Модель с высоким средним F1 не выпускается, если она пропускает критические нарушения, путает пациента или сторону, ошибочно разрешает несопоставимое сравнение либо раскрывает PHI.

## 2. Иерархия решений

Метрики используются на четырёх уровнях.

1. **Hard safety gates:** условия, при нарушении которых релиз запрещён независимо от остальных метрик.
2. **Primary metrics:** одна заранее выбранная основная метрика для каждого клинического вопроса.
3. **Secondary metrics:** объясняют природу качества и помогают выбирать модель.
4. **Monitoring metrics:** обнаруживают ухудшение после внедрения, но сами по себе не являются доказательством клинической эффективности.

### 2.1. Hard safety gates

- отсутствует утечка пациентов, исследований и near-duplicates между split;
- `critical_defect_sensitivity` достигает заранее установленной границы, включая нижнюю границу 95% CI;
- `false_safe_rate` не превышает допустимый предел, включая верхнюю границу 95% CI;
- `patient_pair_error_count = 0` и `laterality_error_count = 0` на интеграционном safety-наборе;
- `false_comparable_rate` имеет ноль наблюдаемых событий на locked safety-наборе;
- неподдерживаемый протокол или неизвестный domain не получает `passed`;
- отклонённое исследование не порождает интерпретируемую динамику;
- отсутствуют подтверждённые PHI leaks, критические уязвимости и неаудируемые изменения результата;
- все результаты воспроизводимы по `datasetVersion`, `modelVersion`, `criteriaVersion`, порогам и commit SHA.

Ноль наблюдаемых ошибок не означает нулевой истинный риск. Для событий с нулём ошибок обязательно публикуется односторонняя верхняя доверительная граница; простое приближение «правило трёх» даёт верхнюю 95% границу около `3 / n`.

## 3. Единицы анализа и наборы оценки

### 3.1. Единицы анализа

Каждая метрика обязана явно указывать уровень:

- `patient` — пациент;
- `study` — исследование;
- `series` — серия;
- `image` — изображение или кадр;
- `site` — анатомическая область и сторона;
- `vertebra` — отдельный позвонок;
- `roi` — область интереса;
- `defect` — тип нарушения;
- `device-day` — аппарат за сутки для phantom/fleet QC;
- `operator-month` — оператор за период для quality improvement.

Повторные исследования одного пациента не считаются независимыми наблюдениями. Confidence intervals и statistical tests кластеризуются на уровне пациента, а для fleet-аналитики дополнительно учитывают site/device/operator.

### 3.2. Обязательные наборы

| Набор | Назначение | Разрешено выбирать пороги |
| --- | --- | --- |
| Train | обучение | да |
| 5-fold patient-grouped OOF | эксперименты, calibration, thresholds | да, только по OOF |
| Locked internal holdout | финальная внутренняя проверка | нет |
| External site holdout | переносимость на другую клинику | нет |
| Leave-one-device/vendor-out | device shift | нет |
| Temporal holdout | изменение протоколов и практики во времени | нет |
| Challenge set | редкие и опасные случаи | нет |
| Prospective silent cohort | реальный workflow без влияния на врача | нет |

Если официальный конкурсный test скрыт, он не используется для выбора архитектуры по частым leaderboard submissions.

### 3.3. Обязательный паспорт любой метрики

Для каждого числа публикуются:

- metric ID и формула;
- версия данных, labelbook, модели и порога;
- единица анализа;
- numerator/denominator;
- число пациентов и число положительных случаев;
- prevalence;
- point estimate и 95% CI;
- missing/abstained/failed cases;
- результаты по обязательным срезам;
- дата расчёта и commit scoring code.

## 4. Итоговое клиническое действие

Целевые действия должны быть богаче, чем `passed/rejected`:

- `accept` — исследование пригодно;
- `review` — требуется эксперт;
- `correct_roi` — исправить анализ без повторного сканирования;
- `reprocess` — повторно обработать исходный кадр;
- `repeat_scan` — повторить получение изображения;
- `exclude_level` — рассмотреть исключение конкретного позвонка;
- `use_alternative_site` — использовать другую область;
- `unsupported` — протокол/domain не поддержан.

### 4.1. Primary metrics

| Метрика | Формула/смысл | Зачем |
| --- | --- | --- |
| `action_macro_f1` | среднее F1 по действиям | не скрывает слабые редкие действия |
| `critical_defect_sensitivity` | `TP / (TP + FN)` для критических дефектов | защищает от опасных пропусков |
| `false_safe_rate` | критически дефектные исследования, ошибочно получившие `accept`, / все критически дефектные | главная safety-метрика |
| `unnecessary_repeat_rate` | пригодные/исправимые исследования, ошибочно отправленные на повтор, / все пригодные/исправимые | контролирует лишнее облучение и потери времени |

### 4.2. Полный набор classification metrics

- sensitivity/recall, specificity, precision/PPV, NPV;
- F1, F2 для критических дефектов и F0.5 для ложных повторов;
- macro/micro/weighted F1;
- balanced accuracy;
- Matthews correlation coefficient;
- Cohen’s kappa и weighted kappa для ordinal severity/action;
- AUROC как threshold-free дополнительная метрика;
- PR-AUC как предпочтительная threshold-free метрика для редких дефектов;
- log loss/NLL для качества вероятностей;
- subset accuracy и Hamming loss для multi-label;
- label ranking average precision;
- confusion matrix в абсолютных числах и долях;
- errors per 100 studies, а не только проценты.

Accuracy и weighted F1 нельзя использовать как единственное доказательство при дисбалансе классов.

## 5. Протокол и маршрутизация

Оцениваются `lumbar_spine_pa`, `hip_left`, `hip_right`, `dual_hip`, `forearm_left/right`, `vfa_lateral`, `whole_body`, `unsupported`.

| Метрика | Уровень | Требование |
| --- | --- | --- |
| `protocol_accuracy` | study | общая правильность |
| `protocol_macro_f1` | study | равный вес протоколов |
| `unsupported_recall` | study | неизвестное не должно попасть в клинический каскад |
| `laterality_accuracy` | site | левая/правая сторона |
| `routing_failure_rate` | study | доля исследований, направленных не той модели |
| `protocol_abstention_rate` | study | доля честных отказов |

Отдельно измеряется `unsafe_routing_rate`: unsupported или неверная сторона, получившие клинический `accept`.

## 6. Сегментация анатомии, ROI и артефактов

### 6.1. Pixel/surface metrics

- Dice: `2|P∩G| / (|P|+|G|)`;
- IoU/Jaccard: `|P∩G| / |P∪G|`;
- class/macro Dice;
- surface Dice при физическом tolerance в миллиметрах;
- Hausdorff distance 95th percentile (`HD95`);
- average symmetric surface distance (`ASSD`);
- boundary F1;
- relative area error;
- centroid distance в мм;
- topology error count;
- число слияний соседних позвонков и пропусков межпозвонкового пространства.

Для ROI одного Dice недостаточно: небольшое смещение границы может слабо изменить Dice, но заметно повлиять на BMD. Поэтому обязательны boundary/surface metrics и влияние на downstream BMD/action.

### 6.2. Instance/object metrics

- AP/AR при заранее заданных IoU thresholds;
- object sensitivity;
- false-positive objects per study;
- vertebra identification accuracy L1–L4;
- artifact localization sensitivity;
- free-response ROC для множественных небольших артефактов;
- correct localization among true-positive classifications.

### 6.3. Клиническая эквивалентность ROI

| Метрика | Смысл |
| --- | --- |
| `roi_area_relative_error` | ошибка площади ROI |
| `roi_boundary_mae_mm` | средняя ошибка границы |
| `bmd_delta_from_roi_error` | изменение BMD из-за отличия ROI |
| `roi_action_agreement` | совпадение решения accept/correct/repeat с экспертом |
| `vertebra_exclusion_agreement` | совпадение правил исключения уровней |

## 7. Landmarks и геометрия позиционирования

- mean/median landmark error в мм;
- normalized mean error (`NME`) относительно ширины/высоты анатомии;
- PCK@2 mm, PCK@5 mm и threshold, согласованный с экспертной воспроизводимостью;
- 95th percentile landmark error;
- failure rate, когда обязательная точка не найдена;
- visibility classification F1;
- angle MAE/RMSE;
- signed angle bias;
- Bland–Altman bias и 95% limits of agreement;
- ICC для измеряемых углов/расстояний;
- spine centerline deviation;
- femoral shaft axis error;
- lesser-trochanter visibility error;
- coverage margin error в мм;
- overlap femoral neck/ischium detection sensitivity;
- repeat-positioning difference baseline/current.

Порог геометрического нарушения задаётся клиническим labelbook и проверяется отдельно по производителю; vendor-specific pixel rules не переносятся между системами без валидации.

## 8. Техническое качество DICOM

### 8.1. Ingestion и decoding

- `study_assembly_success_rate`;
- `dicom_parse_success_rate`;
- `pixel_decode_success_rate`;
- `transfer_syntax_coverage` по частоте реального потока;
- `sop_class_coverage`;
- `multiframe_success_rate`;
- `series_completeness_rate`;
- `metadata_required_field_completeness`;
- `preview_parity` с эталонным viewer;
- `orientation/laterality_parity`;
- `voi_lut_parity`;
- `graceful_unsupported_rate` — неподдерживаемые файлы получают объяснимый отказ, а не ошибочный результат;
- parser crash rate;
- DICOM fuzz-test failures;
- round-trip conformance для создаваемых SEG/SR/GSPS.

Метрики считаются по SOP Class, transfer syntax, manufacturer, model и software version. Актуальная структура DICOM и требования к конформности определяются текущей редакцией стандарта, а не только расширением `.dcm`: [DICOM PS3.10](https://dicom.nema.org/medical/dicom/current/output/html/part10.html) и [DICOM PS3.3](https://dicom.nema.org/medical/dicom/current/output/html/part03.html).

### 8.2. Качество сигнала

- clipped-pixel fraction;
- dynamic range percentile span;
- foreground/background separation;
- motion/blur score и его agreement с экспертом;
- missing anatomy fraction;
- burned-in annotation detection sensitivity/precision;
- external object detection sensitivity;
- corrupt/partial pixel stream recall.

SNR/CNR, PSNR и SSIM допустимы только при наличии валидного reference или контролируемого phantom/noise protocol; их нельзя объявлять клиническим качеством сами по себе.

## 9. Извлечение чисел и Report Auditor

Источники оцениваются отдельно: DICOM SR, private tags, structured export, Secondary Capture, PDF/OCR.

- field exact-match accuracy;
- numeric exact-match accuracy;
- MAE/RMSE по BMD, BMC, area, T-score, Z-score;
- relative numeric error;
- OCR character/word error rate как вспомогательные метрики;
- unit accuracy;
- decimal/sign accuracy;
- laterality/site mapping accuracy;
- reference-database identification accuracy;
- report completeness;
- arithmetic consistency rate;
- clinically material extraction error rate;
- unsupported vendor/template abstention rate.

Главная метрика — не общий OCR accuracy, а доля отчётов без клинически значимой ошибки после полной структуризации.

## 10. Calibration, uncertainty и отказ от ответа

- Brier score;
- negative log-likelihood;
- expected calibration error (`ECE`) с зафиксированным binning;
- adaptive calibration error;
- classwise ECE;
- reliability diagrams;
- calibration slope/intercept;
- risk–coverage curve;
- area under risk–coverage curve (`AURC`);
- selective risk при 80/90/95% coverage;
- coverage при заданном максимальном false-safe risk;
- ensemble disagreement;
- predictive entropy;
- conformal empirical coverage;
- average prediction-set size;
- class-conditional conformal coverage.

Calibration выполняется по OOF/отдельному calibration split и замораживается до holdout. Confidence в UI обязан означать калиброванную вероятность или явно названную uncertainty-меру, а не произвольный score.

## 11. OOD и устойчивость

### 11.1. OOD metrics

- OOD AUROC/AUPR;
- FPR@95%TPR;
- unknown-device recall;
- unsupported-protocol recall;
- OOD-to-review routing rate;
- OOD falsely accepted rate;
- performance drop against in-domain baseline.

### 11.2. Stress tests

Измеряется относительное и абсолютное падение метрик при:

- unseen manufacturer/model/software;
- другой клинике;
- временном holdout;
- изменении окна/контраста;
- допустимом JPEG/export compression;
- шуме и blur;
- небольшом rotation/translation;
- высоком/низком BMI;
- сколиозе и анатомических вариантах;
- металле, протезе и кальцинатах;
- неполном поле;
- burned-in UI;
- редком scan mode.

Недопустимо тестировать robustness на преобразованиях, которые создают анатомически невозможные изображения.

## 12. Longitudinal-метрики

### 12.1. Safety gates сопоставимости

| Метрика | Определение |
| --- | --- |
| `patient_pair_accuracy` | правильный пациент в паре |
| `site_laterality_pair_accuracy` | правильная область и сторона |
| `baseline_selection_accuracy` | выбран допустимый baseline |
| `cross_calibration_gate_sensitivity` | смена аппарата без валидной cross-calibration заблокирована |
| `noncomparability_sensitivity` | критически несопоставимые пары найдены |
| `false_comparable_rate` | несопоставимые пары ошибочно разрешены |

### 12.2. Registration и ROI consistency

- landmark registration error в мм;
- ROI surface Dice после регистрации;
- ROI area-change error;
- vertebra correspondence accuracy;
- positioning difference error;
- registration failure/abstention rate;
- доля случаев, где автоматическая коррекция ROI подтверждена экспертом.

### 12.3. BMD и LSC

- BMD extraction exact-match/MAE;
- абсолютный и процентный bias;
- concordance correlation coefficient;
- ICC;
- Bland–Altman limits of agreement;
- agreement `significant-gain/stable/significant-loss/not-comparable`;
- weighted kappa решения относительно LSC;
- false significant change rate;
- missed significant loss rate;
- new-baseline recommendation agreement;
- facility LSC registry lookup accuracy;
- expired/missing LSC block rate.

ISCD требует facility-specific precision/LSC и запрещает межсистемное количественное сравнение без cross-calibration: [ISCD Official Adult Positions](https://iscd.org/official-positions-2023/).

## 13. Phantom, аппарат и fleet QC

- daily phantom completion rate;
- phantom BMD mean/bias/CV;
- moving-average deviation;
- Shewhart rule violations;
- EWMA/CUSUM alarms;
- drift detection delay;
- false alarm rate;
- time to service acknowledgment/resolution;
- calibration downtime;
- repeat scan rate by device/protocol/operator;
- repeat reason distribution;
- scan rejection and ROI correction rates;
- device/software update performance delta;
- cross-calibration validity coverage;
- studies scanned with expired QC/cross-calibration.

Fleet metrics используются для quality improvement, а не для наказания отдельных операторов без учёта case mix и статистической неопределённости.

## 14. Экспертная разметка

- prevalence и число positive cases каждого класса;
- raw percent agreement;
- Cohen/Fleiss kappa;
- Krippendorff’s alpha для нескольких экспертов и missing labels;
- weighted kappa для severity/action;
- landmark inter-reader error;
- inter-reader Dice/surface Dice;
- adjudication rate;
- uncertain/not-evaluable rate;
- label completeness;
- schema validation failure rate;
- annotation time per study;
- AI pre-label correction time;
- correction distance/area;
- acceptance-without-edit rate;
- disagreement by class/site/device;
- labelbook-version drift.

Экспертная воспроизводимость задаёт практический верхний предел модели. Класс с плохим agreement сначала уточняется в labelbook, а не «лечится» более сложной сетью.

## 15. Качество датасета и защита от утечки

- patient/study/series count;
- class prevalence;
- missingness по каждому полю;
- invalid DICOM rate;
- exact duplicate count;
- near-duplicate rate;
- patient leakage count;
- Study/Series UID leakage count;
- temporal leakage count;
- clinic/device/operator leakage risk;
- train/test distribution divergence;
- sample count и positive count каждого среза;
- vendor/site concentration index;
- label noise estimate;
- unresolved adjudications;
- corrupted/unsupported files;
- percentage with real pixel masks/keypoints versus image-level labels;
- data retention/deletion SLA compliance.

Любой ненулевой подтверждённый leakage count блокирует публикацию holdout-метрик.

## 16. Подгруппы, fairness и worst-group performance

Обязательные срезы при наличии законного и корректного доступа к признакам:

- sex;
- age groups;
- BMI/body thickness;
- manufacturer/model/software;
- site/clinic;
- operator experience;
- left/right;
- protocol/scan mode;
- baseline/follow-up;
- common/rare defect;
- металл/протез/сколиоз/дегенеративные изменения;
- image quality and OOD status.

Для каждого среза публикуются sensitivity, specificity, PPV, NPV, F1, calibration и abstention. Дополнительно:

- worst-group sensitivity/F1;
- max–min performance gap;
- relative error-rate ratio;
- calibration gap;
- abstention gap;
- false-safe gap.

Малые подгруппы не скрываются, но помечаются широким CI. Нельзя интерпретировать случайные различия как bias без достаточной мощности.

## 17. Клиническая полезность и reader study

- технические ошибки, пропущенные человеком без AI;
- ошибки, пропущенные human+AI;
- sensitivity/specificity врача и human+AI;
- время проверки исследования;
- время от сканирования до исправления;
- доля исправлений до ухода пациента;
- repeat scan rate до/после;
- ненужные повторы;
- ROI correction rate;
- override rate и причины;
- alert acceptance rate;
- automation-bias errors;
- inter-reader agreement;
- diagnostic/report completeness, если модуль отчёта включён;
- decision-curve net benefit;
- number needed to review/correct;
- adverse event/near-miss count.

Дизайн reader study должен быть paired/crossover с washout и балансировкой порядка, а статистика — кластеризованной по врачу и пациенту. Международные принципы прозрачности требуют оценивать производительность human-AI team, а не только изолированной модели: [FDA ML-enabled device transparency](https://www.fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles).

## 18. Usability и доступность

- task completion rate;
- critical use-error rate;
- median time on task;
- число кликов/возвратов для исправления;
- ошибочное подтверждение critical alert;
- alert comprehension accuracy;
- explanation comprehension;
- SUS score;
- NASA-TLX workload;
- обучение до самостоятельной работы;
- keyboard-only task success;
- screen-reader task success;
- WCAG automated violations и manual audit findings;
- localization defects;
- contrast/focus/touch-target violations;
- support requests per 100 users.

В usability study обязательно участвуют технологи, врачи, медицинские физики и администраторы, потому что у них разные задачи и риски.

## 19. Производительность и надёжность

- end-to-end latency p50/p90/p95/p99;
- decode, preprocess, inference, postprocess и report latency отдельно;
- queue wait time;
- studies/hour и concurrent studies;
- cold-start latency;
- CPU/GPU/RAM/VRAM peak;
- model artifact size;
- timeout rate;
- inference failure rate;
- retry rate;
- availability/error-budget burn;
- mean time to detect/acknowledge/recover;
- data loss rate;
- backup success, RPO и RTO;
- offline demo success rate;
- cost per study и infrastructure cost per site.

Latency публикуется вместе с hardware, batch size, image resolution и transfer syntax.

## 20. MLOps и post-deployment monitoring

- reproducible-run rate;
- test pass rate и flaky-test rate;
- model/data/criteria version coverage в результатах;
- lineage completeness;
- training failure rate;
- experiment-to-production lead time;
- rollback success/time;
- shadow/canary disagreement;
- prediction distribution drift;
- input embedding drift (MMD/energy distance);
- PSI/Jensen–Shannon divergence для контролируемых признаков;
- class prevalence drift;
- calibration drift;
- abstention/OOD drift;
- device/site mix drift;
- delayed-label performance;
- incident count by severity;
- retraining trigger precision/recall;
- percentage traffic on validated model versions.

Drift-сигнал не запускает автоматическое самообучение. Он запускает расследование, разметку, независимую валидацию и управляемый релиз.

## 21. Privacy, security и auditability

- direct-tag PHI removal recall/precision;
- private-tag PHI recall;
- burned-in PHI OCR recall;
- `phi_leak_count` в export/log/cache;
- pseudonym collision rate;
- unauthorized access attempts/successes;
- RBAC policy test coverage;
- audit-event completeness;
- immutable result/version coverage;
- secrets detected in repository/artifacts;
- critical/high vulnerabilities;
- patch SLA compliance;
- SBOM coverage;
- dependency/license policy violations;
- encryption coverage at rest/in transit;
- backup restore test success;
- retention/deletion request compliance;
- penetration-test findings and closure time.

Поле `Patient Identity Removed` не считается доказательством полного обезличивания; отдельно проверяются private tags и Pixel Data.

## 22. Бизнес- и системный эффект

Эти метрики не заменяют clinical safety, но показывают ценность продукта:

- снижение repeat scan rate;
- сокращение времени контроля/отчёта;
- studies processed per technologist;
- стоимость исправленной ошибки;
- cost per accepted study;
- экономия времени врача/оператора;
- доля исправлений до ухода пациента;
- deployment lead time в новой клинике;
- vendor integration coverage;
- active users/sites/devices;
- adoption и feature utilization;
- renewal/retention;
- support burden;
- ROI/payback period;
- снижение вариабельности качества между операторами и аппаратами.

## 23. Статистический протокол

1. Primary endpoints и non-inferiority/safety margins фиксируются до открытия holdout.
2. Пороги выбираются только по patient-grouped OOF.
3. Для proportions используется Wilson/exact interval; для сложных метрик — patient-cluster bootstrap не менее 10 000 повторов.
4. Для сравнения моделей используются paired predictions: McNemar для бинарных решений и paired bootstrap для F1/PR-AUC/Dice.
5. Публикуются effect size и CI, а не только p-value.
6. Для множества exploratory slices применяется Benjamini–Hochberg FDR; заранее заданные safety endpoints не смешиваются с exploratory анализом.
7. Sample size рассчитывается по желаемой ширине CI и prevalence критического класса.
8. Отсутствующие/неоценимые случаи показываются отдельно; их нельзя молча удалять.
9. Все abstentions входят в coverage и workflow analysis.
10. Анализ выполняется на уровне пациента и повторяется по study/site/device slices.

## 24. Release gates по стадиям

### Конкурсный релиз

- scoring code совпадает с организатором;
- patient-grouped OOF и locked holdout;
- нулевой leakage audit;
- per-class metrics и confusion matrices;
- calibration и frozen thresholds;
- реальный overlay, provenance и offline demo;
- latency/error benchmark;
- error atlas критических FN/FP.

### Silent clinical pilot

- пройдены hard safety gates;
- server-side DICOM matrix покрывает поток клиники;
- external/device holdout;
- privacy/security test pack;
- ручное подтверждение и audit trail;
- мониторинг drift/abstention/failures;
- утверждённый intended use и risk register.

### Мировой multi-site продукт

- независимая multi-site prospective validation;
- reader study human versus human+AI;
- worst-group и vendor-shift evidence;
- phantom/fleet QC;
- post-market monitoring и управляемый rollback;
- документация total product lifecycle и применимых стандартов;
- доказанный clinical/operational benefit без ухудшения safety endpoints.

## 25. Минимальные dashboard

1. **Competition:** официальный score, macro F1, per-class PR-AUC, folds, leaderboard history.
2. **Safety:** critical sensitivity, false-safe, false-repeat, unsafe routing, false-comparable.
3. **Anatomy:** Dice/surface Dice, landmark mm error, ROI/BMD impact.
4. **Calibration/OOD:** reliability, risk–coverage, unknown device/protocol.
5. **Longitudinal:** pair matching, comparability, registration, LSC decisions.
6. **Data/labels:** prevalence, agreement, missingness, duplicates, leakage.
7. **Operations:** latency, failures, queue, uptime, cost.
8. **Fleet:** repeat reasons, operator/device variation, phantom drift.
9. **Security/privacy:** PHI tests, vulnerabilities, access/audit events.
10. **Clinical value:** time, corrections before departure, repeat reduction, overrides.

## 26. Machine-readable registry

Ключевые primary и safety metrics продублированы в `docs/metrics.registry.json`. Evaluation pipeline должен читать этот файл, проверять наличие обязательных метрик и запрещать release, если отсутствует значение, denominator, CI или обязательный slice.

