"""Generate current slides and release facts from measured, matching artifacts.

Historical experiment reports stay immutable. Current materials explicitly
report incomplete requirements instead of presenting CSV validity as readiness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from html import escape
from pathlib import Path

from dxaqc import __version__
from dxaqc.decision import VERSION


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_facts(metrics_path, oof_path, batch, repeat, image_id, quality_review_evaluation=None,
                  workflow_profile_path=None):
    metrics = json.loads(metrics_path.read_text())
    allowed_versions = {VERSION, 'review-1'}
    versions = metrics.get('decision_versions', [])
    if not versions or not set(versions) <= allowed_versions or metrics['predictions_sha256'] != digest(oof_path):
        raise ValueError('Metrics must match the current decision version and OOF file')
    with oof_path.open(newline='') as stream:
        oof = list(csv.DictReader(stream))
    if len(oof) != metrics['labelled_images'] or sorted({row.get('decision_version') for row in oof}) != sorted(versions):
        raise ValueError('OOF rows must match the current decision version')
    first = json.loads((batch / 'summary.json').read_text())
    second = json.loads((repeat / 'summary.json').read_text())
    if first['version'] != __version__ or second['version'] != __version__:
        raise ValueError('Batch version differs from current release')
    workflow = first.get('workflow_profile_id')
    if workflow != second.get('workflow_profile_id'):
        raise ValueError('Repeated batch uses a different workflow profile')
    profile = None
    if workflow:
        profile_path = workflow_profile_path or Path(__file__).parent / (
            'models/workflow_' + __version__.replace('.', '_') + '/profile.json')
        profile = json.loads(profile_path.read_text())
        if profile['profile_id'] != workflow or quality_review_evaluation is None or 'review-1' not in versions:
            raise ValueError('Workflow profile requires its matching quality review evaluation; baseline metrics are insufficient')
        evaluation = json.loads(quality_review_evaluation.read_text())
        if (evaluation['model_sha256'] != profile['artifacts']['DXAQC_QUALITY_REVIEW_MODEL']['sha256']
                or evaluation['metrics'] != metrics):
            raise ValueError('Quality review weights or metrics differ from the supplied evaluation')
    paths = [metrics_path, oof_path, *[root / name for root in (batch, repeat)
             for name in ('summary.json', 'requirements.json', 'timing.json',
                          'submission.csv', 'results_extended.csv', 'additional_series.zip',
                          'submission_validation.json', 'full_validation.json')]]
    validations = [json.loads((root / 'full_validation.json').read_text()) for root in (batch, repeat)]
    if any(not value['valid'] for value in validations):
        raise ValueError('Manifest, repeat, timing or series validation failed')
    if not image_id.startswith('sha256:') or len(image_id) != 71:
        raise ValueError('Exact container image ID is required')
    return {'version': __version__, 'decision_version': '+'.join(versions), 'image_id': image_id,
            'workflow_profile': profile,
            'metrics_scope': 'Historical internal OOF of quality/type models; not an independent test of projection, ROI rules or this delivery',
            'weights_sha256': digest(Path(__file__).parent / 'models/bundle.joblib'),
            'code_sha256': {str(path.relative_to(Path(__file__).parent.parent)): digest(path)
                            for path in sorted((Path(__file__).parent / 'dxaqc').glob('*.py'))},
            'metrics': metrics, 'batch': first, 'repeat': second,
            'requirements': json.loads((batch / 'requirements.json').read_text()),
            'timing': json.loads((batch / 'timing.json').read_text()),
            'validation': validations,
            'series_repeat_exact_match': digest(batch / 'additional_series.zip') == digest(repeat / 'additional_series.zip'),
            'artifacts': {str(path): digest(path) for path in paths},
            'scope': 'local offline engineering evidence and internal OOF; not clinical acceptance'}


def presentation(facts):
    m, batch, requirements = facts['metrics'], facts['batch'], facts['requirements']
    overall = m['overall_quality']
    def number(value):
        return f'{value:.3f}' if value is not None else 'не определено'
    def items(values):
        return '<ul>' + ''.join('<li>' + escape(str(v)) + '</li>' for v in values) + '</ul>'
    metric_rows = ''
    for key, label in (('f1', 'F1'), ('roc_auc', 'ROC-AUC'), ('sensitivity', 'Чувствительность'),
                       ('specificity', 'Специфичность'), ('average_precision', 'Average precision')):
        interval = overall['ci95_study_bootstrap'][key]
        metric_rows += f'<tr><th scope="row">{label}</th><td>{number(overall[key])}</td><td>{number(interval["low"])}–{number(interval["high"])}</td></tr>'
    criterion_rows = ''.join(f'<tr><th scope="row">{escape(key)}</th><td>{number(value["f1"])}</td><td>{value["tn_fp_fn_tp"][3]} / {value["tn_fp_fn_tp"][2]} / {value["tn_fp_fn_tp"][1]}</td></tr>'
                             for key, value in m['by_criterion'].items())
    sections = [
        ('Osseo AI — контроль качества DXA', f'<p>Релиз {escape(facts["version"])} · решение v{escape(facts["decision_version"])}</p><p>Локальная пакетная обработка позвоночника и бедра. Полное выполнение анатомических требований пока не подтверждено.</p>'),
        ('Задача и архитектура', items(['DICOM/ZIP → декодирование и масштаб Y=1,05/X=0,6 мм → роутер области → геометрия и ResNet18 → RF/LR по критериям → CSV/XLSX и SC/SR.',
                                      'Три официальных типа позвоночника, два типа бедра; несколько нарушений разделяются «;».',
                                      'Вторичный пересмотр может определить тип либо отменить тревогу; чувствительность оценивается по итоговому решению.',
                                      'Failure — технический отказ с пустыми классом и вероятностью, отдельно от медицинского результата.'])),
        ('Данные и оценка', items([f'{m["labelled_images"]} размеченных уникальных изображений, {m["studies"]} исследований; 499 файлов учебного экспорта включают копии.',
                                  'Пять внешних фолдов по исследованиям, один повтор; пороги выбраны во внутренних обучающих фолдах.',
                                  'Синтетические повороты используются только в обучении вместе с родительским исследованием.',
                                  'Это внутренний OOF на ранее изученных данных. Независимость пациентов между исследованиями не доказана; три тестовых DICOM не имеют ответов.'])),
        ('Метрики внутренней ретроспективной оценки', '<table><caption>Ранее изученный внутренний OOF; 95% ДИ — bootstrap по исследованиям. Не независимая оценка новой поставки.</caption><tr><th>Метрика</th><th>Значение</th><th>95% ДИ</th></tr>' + metric_rows + '</table>' + f'<p>Macro-F1 типов: {number(m["criterion_macro_f1"])}. Положительных ответов без типа: {m["predicted_positive_without_type"]}.</p>'),
        ('Ошибки по критериям', '<table><caption>TP / FN / FP по текущему OOF</caption><tr><th>Критерий</th><th>F1</th><th>TP / FN / FP</th></tr>' + criterion_rows + '</table><p>Особое внимание — оси позвоночника, ротации бедра и малому числу положительных ROI. Порог оси остаётся 5° по ТЗ.</p>'),
        ('Офлайн-поставка и время', items([f'Повторные офлайн-прогоны контейнерного CLI с одним профилем: {batch["success"]}/{batch["files"]} Success в первом пакете.',
                                                f'Весь первый пакет с таблицами и SC/SR: {facts["timing"]["batch_seconds"]:.3f} с. Консервативная верхняя граница на исследование: {facts["timing"]["max_study_upper_bound_seconds"]:.3f} с.',
                                                f'Пик RSS процесса: {batch["process_peak_rss_bytes"] / 1024**2:.1f} МиБ. GPU не используется.',
                                                'Сопоставлены независимый список входов, повтор предсказаний, ссылки SC/SR и полное время. Это локальный CPU-хост, не официальный H200-стенд.',
                                                'Рекомендуемый старт: 8 CPU, 8 ГиБ RAM, 40 ГиБ диска; GPU не требуется.'])),
        ('Что ещё не соответствует', items([f'Незавершённая типизация в полном выводе: {requirements["untyped_violations"]} строк.',
                                                       'Th12/L1–L4 и анатомические зоны не имеют подтверждённого локализатора. Проекция позвоночника проверяется до QC; проекция бедра и клиническая точность не подтверждены.',
                                                       'Исходной ROI в данных организатора нет. Для поддерживаемой ROI сохраняется назначение; правила охвата не применяются к ROI шейки.',
                                                       'Нет независимого врачебного эталона, пациентского разделения и измерения на H200. Это ограничения, а не завершённые проверки.'])),
        ('Демонстрация и следующий этап', items(['Загрузить DICOM → открыть исходное изображение → показать критерии, угол и источник масштаба.',
                                                            'Показать неопределённые анатомические проверки; открыть требования и протокол времени.',
                                                            'Добавить геометрию охвата, проверить отступы, подтвердить допустимый перенос, сохранить отдельное решение специалиста.',
                                                            'До клинического пилота: независимая разметка ориентиров и проекции, определение назначения ROI, обучение и проверка на новом наборе.',
                                                            'Сравнения прежних архитектур доступны в архиве competition/reports; их метрики не подписываются текущей версией.'])),
    ]
    css = 'body{margin:0;background:#eef2f5;color:#182736;font:22px/1.5 system-ui,sans-serif}main{max-width:1120px;margin:auto}.slide{background:white;padding:50px 60px;margin:24px 0;min-height:650px;box-sizing:border-box}h1{font-size:44px;line-height:1.15}li{margin:14px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:10px;text-align:left;border-bottom:1px solid #cad5df}caption{text-align:left;font-size:17px}.num{font-size:16px;color:#516778}@media(max-width:700px){body{font-size:18px}.slide{padding:24px;min-height:0}h1{font-size:32px}}@media print{@page{size:A4 landscape;margin:0}body{background:white;font-size:16px}.slide{margin:0;width:297mm;min-height:210mm;padding:14mm 18mm;break-after:page}h1{font-size:30px}.slide:last-child{break-after:auto}}'
    body = ''.join(f'<section class="slide" aria-labelledby="s{i}"><p class="num">{i} / {len(sections)}</p><h1 id="s{i}">{escape(title)}</h1>{content}</section>' for i, (title, content) in enumerate(sections, 1))
    return f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Osseo AI — текущая поставка</title><style>{css}</style></head><body><main>{body}</main></body></html>\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metrics', 'oof', 'batch', 'repeat', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--quality-review-evaluation', type=Path,
                        help='evaluation report matching the shipped secondary model')
    parser.add_argument('--workflow-profile', type=Path,
                        help='exact shipped profile; defaults to the current versioned profile')
    args = parser.parse_args()
    facts = release_facts(args.metrics, args.oof, args.batch, args.repeat, args.image_id,
                          args.quality_review_evaluation, args.workflow_profile)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'CURRENT_RELEASE.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'FINAL_PRESENTATION_RU.html').write_text(presentation(facts))
    print(json.dumps({'version': __version__, 'requirements_complete': facts['requirements']['complete']}))


if __name__ == '__main__':
    main()
