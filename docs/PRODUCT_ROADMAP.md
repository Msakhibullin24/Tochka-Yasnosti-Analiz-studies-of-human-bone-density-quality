# Полный аудит и roadmap продукта

## Итог аудита

Прототип уже убедительно демонстрирует клинический workflow: реальный локальный DICOM ingest, privacy/provenance, понятные нарушения, безопасный отказ от анатомических утверждений для загруженного файла и анализ в динамике с LSC-gate. Главный разрыв до «лучшего решения России» находится не в косметике, а в данных, доказательствах метрик, реальном ML overlay, полноте DICOM и интеграции с контуром клиники.

## P0 — обязательно для конкурса

| Работа | Результат | Проверка готовности |
| --- | --- | --- |
| Зафиксировать официальную метрику и формат submission | evaluation harness одной командой | локальный score совпадает с организатором |
| Labelbook + двойная разметка | воспроизводимые классы дефектов | agreement по каждому классу и adjudication |
| Patient-grouped folds | отсутствие утечки | UID/hash/near-duplicate audit проходит |
| Multi-task baseline | segmentation/keypoints + multi-label QC | полный OOF, а не один random split |
| Calibration и abstention | честные `passed/review/rejected` | ECE/Brier и risk–coverage на holdout |
| Реальные overlays | mask/keypoints конкретной модели | pixel-to-result traceability |
| Longitudinal quality gate | запрет ложного сравнения | тесты смены аппарата, ROI и плохой укладки |
| Offline demo | стабильный 90-секундный сценарий | повторяется на чистой машине без сети |

## P1 — максимум конкурсных метрик

- OOF error ledger: сначала false negative критических дефектов, затем ложные повторы.
- Hard-negative mining и active learning на кейсах с расхождением экспертов/ансамбля.
- Device/site slices и leave-one-device-out stress-test.
- Domain-aware normalization без использования производителя как shortcut.
- Fold ensemble только после ablation, если прирост устойчив на OOF и holdout.
- Class-specific thresholds под официальную функцию потерь.
- Patient-level bootstrap 95% CI, latency p50/p95, peak memory и failure rate.
- Отдельный challenge set: металл, кальцинаты, протезы, обрезанный охват, burned-in UI, нестандартный экспорт.

## P1 — продукт и интеграция

- backend job queue, object storage с TTL, idempotency и request ID;
- DCMTK/GDCM decoder для JPEG-LS, JPEG 2000, RLE и multiframe;
- DICOMweb `STOW-RS/WADO-RS/QIDO-RS`, PACS/RIS sandbox;
- ручное исправление ROI, approve/reject и полный audit trail;
- PDF/JSON отчёт, Structured Report/FHIR mapping после согласования с клиникой;
- RBAC, журнал доступа, шифрование, backup/restore и threat model;
- data/model/criteria/LSC registry с immutable версиями;
- наблюдаемость без PHI: latency, failures, abstention, device drift, class drift.

## P2 — клинический пилот

- prospective silent mode: модель работает, но не влияет на решение;
- reader study: время врача, agreement, пропущенные дефекты, ненужные повторы;
- controlled rollout по аппаратам и протоколам;
- post-deployment monitoring и автоматический rollback модели;
- анализ пользы: снижение повторов, времени проверки и доли некорректных исследований;
- регуляторная стратегия, менеджмент риска, usability и независимая ИБ-проверка.

## Метрики верхнего уровня

| Контур | Метрика |
| --- | --- |
| Конкурс | официальная метрика + macro F1 типов нарушений |
| Безопасность | sensitivity критических дефектов, false-safe rate |
| Разметка | Dice/surface Dice, HD95, PCK/NME |
| Доверие | ECE, Brier, risk–coverage |
| Динамика | sensitivity несопоставимости, agreement решения относительно LSC |
| Производительность | p50/p95 latency, failure rate, peak memory |
| Клиника | repeat rate, review time, override rate, net benefit |

## Следующие пять конкретных действий

1. Получить структуру датасета и официальный scoring code; сразу построить leakage audit.
2. Провести 100–200 кейсов через pilot labelbook с двумя экспертами и исправить неоднозначные правила.
3. Обучить простые baseline-модели и зафиксировать OOF до выбора сложного backbone.
4. Подключить реальный inference к существующему `Study`-контракту и заменить demo overlays.
5. Собрать evidence pack: reproducibility script, model/data cards, CI, error atlas, latency и видео offline demo.

## Что нельзя обещать раньше времени

- «98% точности» без patient-level holdout, prevalence и confidence intervals;
- извлечение BMD из любого DICOM без матрицы поддержанных производителей;
- сравнение разных аппаратов без cross-calibration;
- полное обезличивание только по одному DICOM-тегу;
- диагноз или замену врача;
- клиническую готовность до независимой валидации и применимого регуляторного процесса.
