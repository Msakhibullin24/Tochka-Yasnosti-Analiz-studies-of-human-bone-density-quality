# Независимые модели DXA QC

Версия 1.9.3. Модели в этом наборе **не объединяют признаки, баллы или решения**. У каждой собственный TorchScript `model.ts` с энкодером и головой, свой `model.json` с порогами и SHA-256. Основной конкурсный класс и типы нарушений по-прежнему выдаёт отдельный `dxaqc` bundle. Оценки ниже имеют `mode=shadow` и не меняют его результат.

Актуальное парное сравнение на общем test из 46 DXA и 19 исследований, включая типы нарушений и 95% интервалы разности: [аудит независимых моделей](../../docs/INDEPENDENT_SPECIALISTS_PAIRED_AUDIT_2026_09_24_RU.md). Региональные модели дали F1 0,462 против 0,432 у train-only гибрида, но AUC 0,639 против 0,869; нынешний `typed_v4` по другому протоколу на этих строках имеет F1 0,571. Продвижение shadow-моделей в основной вердикт по этим данным не обосновано.

| ID | Зона ответственности | Выход |
| --- | --- | --- |
| `spine_convnext` | AP DXA позвоночника, охват, ось и артефакты | Собственные баллы и пороги `quality`, `spine_coverage`, `spine_axis`, `spine_artifact` |
| `hip_convnext` | AP DXA бедра, позиционирование и ROI | Собственные баллы и пороги `quality`, `hip_position_rotation`, `hip_roi_coverage` |
| `general_efficientnet` | Независимая общая проверка обоих регионов | Собственные баллы и пороги применимых критериев |

Ранее сохранённые внутренние test-метрики каждой модели относятся к **разным подвыборкам** и не служат сравнением друг с другом: spine ConvNeXt — 19 снимков / 7 положительных, F1 0,571; hip ConvNeXt — 27 / 3, F1 0,333; EfficientNet — 46 / 10, F1 0,372. Протокол, AUC и ссылки на отчёты указаны в [машинной сводке](../../docs/competition/independent_portfolio_2026_09_24.json). Эти числа не равны метрикам основного QC на 249 OOF-снимках и не являются внешней клинической проверкой.

Для снимка другой области модель возвращает `not_applicable`. Если одна модель не смогла выполнить inference, только её запись получает `unavailable`; остальные результаты и основной вердикт сохраняются. Расширенный CSV содержит отдельный JSON-объект `specialist_outputs`, API detail и веб-карточка показывают каждый ID отдельным блоком. Строгий конкурсный CSV не содержит этих экспериментальных полей.

## Упаковка существующих обученных моделей

```bash
PYTHONPATH=competition competition/.venv/bin/python competition/pack_independent_specialists.py \
  --spine data/specialists/runs/convnextv2-spine-finetune-v3 \
  --hip data/specialists/runs/convnextv2-hip-finetune-v3 \
  --general data/specialists/runs/efficientnet-b4-bce-v2 \
  --output data/specialists/runs/independent-portfolio-v1
```

Упаковка проверяет исходные checkpoint и совпадение версии разметки, копирует каждый `model.ts` отдельно и фиксирует контрольные суммы. Подготовленный каталог находится в исключённом из Git `data/`; `git push` **не перенесёт веса** на другой ПК. Для копирования подготовлен архив `data/specialists/exports/independent-portfolio-v1.tar` (295 567 360 байт) и соседний `.tar.sha256`. После переноса архива и checksum-файла в тот же каталог на новом ПК:

```bash
cd data/specialists/exports
sha256sum -c independent-portfolio-v1.tar.sha256
mkdir -p ../runs
tar -xf independent-portfolio-v1.tar -C ../runs
```

Локальный пакетный запуск:

```bash
DXAQC_SPECIALIST_PORTFOLIO=data/specialists/runs/independent-portfolio-v1 \
PYTHONPATH=competition competition/.venv/bin/python -m dxaqc.cli \
  --input /path/to/dicom --output /path/to/new-output --no-explanations
```

В Docker Compose:

```bash
docker compose -f docker-compose.yml -f docker-compose.independent-specialists.yml up --build
```

Для старой одиночной модели остаётся `DXAQC_SPECIALIST_PATH`. Одновременно задавать две переменные нельзя: система отклонит конфигурацию, чтобы не смешать форматы результатов. Подробности старого режима — в [SPECIALISTS.md](SPECIALISTS.md).

Тренер отдельной модели поддерживает `--device cpu|cuda|auto` (по умолчанию `cpu`).
На RTX 5090 используйте совместимое CUDA-окружение и `--device cuda`; обучение на этой
машине ещё не измерено. Сохранённый `model.ts` переносится на CPU перед экспортом,
поэтому конкурсный офлайн-инференс не зависит от наличия GPU. [Инструкция для GPU](../../docs/RTX_5090_TRAINING_RUNBOOK_RU.md).

## Другие загруженные модели

Три DAX энкодера запускаются **по одному** через [`run_dxa_candidate.py`](../run_dxa_candidate.py). Каждый записывает собственный вектор признаков `.npy`; на одном DXA получены отдельные конечные векторы размерности 512, 192 и 192. Их study-held-out F1 оказался ниже основного QC, поэтому классификатор из них в конкурсный вывод не включён. LiteMedSAM отдельно выдаёт маску по заданному прямоугольнику и требует проверки экспертом. DXA-to-3D теперь запускается для `--protocol spine` через [`analyze_specialists.py`](../analyze_specialists.py) и отдельно выдаёт `dxa-to-3d-raw.npy`; соответствие его координат официальным ориентирам пока не установлено. Каждый выход также получает собственный `<model_id>.json`. Общие COCO-детекторы отключены по умолчанию и доступны только с явным `--include-generic-coco` для исследования. Ни один из этих результатов не усредняется с тремя QC-специалистами.

Локальная и офлайн-контейнерная технические проверки набора на трёх DICOM без ответов: 3/3 Success, strict submission valid=true; основные прогнозы совпали с запуском без портфеля. Полный офлайн-прогон на обучающем пакете: **499/499 Success**, 100 исследований, strict valid=true, независимый manifest и повторная сверка основного результата прошли. `spine_convnext` сработал на 166 снимках, `hip_convnext` — на 333, `general_efficientnet` — на всех 499; чужая область получила `not_applicable`. Среднее время 0,193 с на снимок, максимум 4,476 с на исследование, peak RSS около 1 ГБ. Это техническая проверка, а не оценка точности на независимых метках. Модельные баллы не следует представлять как клинически подтверждённые вероятности. [Машинная сводка](../../docs/competition/independent_portfolio_2026_09_24.json), [strict](../reports/independent_portfolio_submission_validation.json), [manifest и повтор](../reports/independent_portfolio_extended_validation.json).
