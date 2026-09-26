# Датасеты

Данные лежат архивами, чтобы репозиторий оставался компактным, а целостность проверялась: `sha256sum -c SHA256SUMS`.
Репозиторий приватный. **Не публикуйте эту папку**: `organizer/` — обезличенные медицинские изображения организатора
конкурса, их распространение регулируется правилами конкурса.

| Архив | Что внутри | Как используется |
|---|---|---|
| `organizer/НД_для_обучения.zip` | 499 DICOM (100 исследований, 252 уникальных изображения) + `разметка.xlsx` | обучение и валидация |
| `organizer/Для теста.zip` | 3 DICOM для проверки формата | отладка загрузчика, тест `DXAQC_DEBUG_SET` |
| `external/ramathibodi/osteoporosis.zip` | открытый набор DXA Hologic с экспертными метками качества (CC BY, см. `osteoporosis.provenance.json`) | только внешняя проверка: `competition/external_validation.py` |
| `external/hipray.zip` | 139 рентгенограмм бедра с масками (CC BY 4.0) | не используется моделью; оставлен для работ по ориентирам |

## Переобучение с нуля
```sh
unzip datasets/organizer/НД_для_обучения.zip -d /tmp/dxa
cd competition
python train.py --dataset /tmp/dxa/Исследования --synthetic rotate     # модель + reports/validation_metrics.json
python leakage_check.py --dataset /tmp/dxa/Исследования                # негативный контроль
```
Имена в архиве записаны в кодировке UTF-8; если `unzip` показывает «кракозябры», используйте
`python -c "import sys; sys.path.insert(0,'competition'); from pathlib import Path; from dxaqc.pipeline import safe_extract; safe_extract(Path('datasets/organizer/НД_для_обучения.zip'), Path('/tmp/dxa'))"`.

Актуальные внешние данные находятся в `data/external/`: [перечень, проверки и загрузки](../docs/DATASETS_FOR_TZ_2026_09_26_RU.md).
