"""Generate the anatomy/QC pitch from measured audit artifacts.

Keeps nested validation separate from full-data fitting and runtime acceptance.
Does not overwrite the CPU release passport or imply complete TZ compliance.
"""
from __future__ import annotations
import argparse
from html import escape
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('docs/competition/anatomy_upgrade_2026_10_15.json'))
    parser.add_argument('--output', type=Path, default=Path('docs/FINAL_PRESENTATION_RU.html'))
    args = parser.parse_args()
    r = json.loads(args.report.read_text())
    candidate = next(t for t in r['trials'] if t['name'] == 'quality-gated-own-roi')['metrics']
    v = r['verification']['gpu_fullfit_1_11_7']
    cli, http, cold = v['cli'], v['http'], v['deployment_protocol']
    if not http['cli_answers_identical'] or cli['success'] != 499 or http['success'] != 499:
        raise ValueError('Full offline CLI and HTTP acceptance is required before generating current slides')
    q = candidate['overall_quality']
    f1ci = q['ci95_study_bootstrap']['f1']
    auc_ci = q['ci95_study_bootstrap']['roc_auc']
    roi = r['native_noise_neck_adaptation']
    cover = v['regression_against_prior_fullfit']
    def n(value): return f'{value:.3f}'.replace('.', ',')
    def items(values): return '<ul>' + ''.join('<li>' + escape(str(s)) + '</li>' for s in values) + '</ul>'
    def table(caption, headers, rows):
        return ('<table><caption>' + escape(caption) + '</caption><tr>' + ''.join('<th scope="col">' + escape(s) + '</th>' for s in headers)
                + '</tr>' + ''.join('<tr>' + ''.join(('<th scope="row">' if j == 0 else '<td>') + escape(str(cell))
                + ('</th>' if j == 0 else '</td>') for j, cell in enumerate(row)) + '</tr>' for row in rows) + '</table>')
    labels = {'spine_coverage': 'Укладка позвоночника', 'spine_axis': 'Ось позвоночника',
              'spine_artifact': 'Посторонние предметы', 'hip_position_rotation': 'Укладка бедра', 'hip_roi_coverage': 'ROI бедра'}
    criterion = [(labels[name], n(m['f1']), f"{m['tn_fp_fn_tp'][3]} / {m['tn_fp_fn_tp'][2]} / {m['tn_fp_fn_tp'][1]}")
                 for name, m in candidate['by_criterion'].items()]
    sections = [
        ('Osseo AI — контроль качества DXA', '<p>Техническая сборка 1.11.7 · отдельный GPU-кандидат</p>' + items([
            'Две анатомические области, пять типов нарушений, CSV/XLSX, визуальная серия и DICOM SR.',
            'Исследовательские модели обучены и проверены офлайн. Полное выполнение ТЗ пока не заявляется.'])),
        ('Работающий маршрут', items([
            'DICOM/ZIP → область → исходная калибровка Y=1,05/X=0,6 мм → DINOv3 + RAD-DINO и геометрия → общий QC и типы.',
            'TSXR: нумерованные маски позвоночника и бедра; сети ориентиров и собственные ROI в исходных координатах.',
            'Ось L1–L4 измеряется в физических координатах; правило больше 5° действует независимо от вероятностного порога.',
            'Более точная разметка отделена от фиксированных входов проверенных голов QC.'])),
        ('Данные и протокол', items([
            f"249 размеченных изображений / 100 групп исследований; {r['qc_trials_completed']} дополнительных вариантов QC после первых трёх.",
            'Пять внешних и четыре внутренних разбиения по исследованиям. Пациентская независимость не установлена.',
            '499 входных DICOM содержат 252 уникальных вычисления. Три отдельных отладочных DICOM не имеют ответов.',
            'Когорта уже многократно исследована. Поисковое сравнение не является новым независимым тестом.',
            'Финальные головы обучены на всех 249 метках; их обучающая точность не выдаётся за результат проверки.'])),
        ('Метрики выбранного QC-кандидата', table('Вложенная внутренняя проверка; 95% ДИ — бутстрэп исследований.',
            ['Показатель', 'Исходное сравнение', 'Кандидат'], [
                ['F1', n(r['baseline']['f1']), f"{n(q['f1'])} ({n(f1ci['low'])}–{n(f1ci['high'])})"],
                ['ROC-AUC', n(r['baseline']['roc_auc']), f"{n(q['roc_auc'])} ({n(auc_ci['low'])}–{n(auc_ci['high'])})"],
                ['Чувствительность', n(r['baseline']['sensitivity']), n(q['sensitivity'])],
                ['Специфичность', n(r['baseline']['specificity']), n(q['specificity'])],
                ['Average precision', n(r['baseline']['average_precision']), n(q['average_precision'])]])
            + '<p>Кандидат пропустил 21 нарушение вместо 28, дал 43 ложные тревоги вместо 34. Это компромисс, а не рост всех показателей.</p>'),
        ('Типы нарушений', table('Тот же вариант с наибольшим F1 QC; TP / FN / FP.', ['Критерий', 'F1', 'TP / FN / FP'], criterion)
            + f'<p>Средний F1 типов: {n(candidate["criterion_macro_f1"])}. Другой вариант достигает 0,597 при общем F1 0,603.</p>'),
        ('ROI и анатомическая разметка', items([
            f"Собственные ROI бедра: {cover['hip_roi_proposed']}/{cover['hip_unique_images']}. Один отказ локализации сохранён явно.",
            f"Позвоночник: {cover['spine_roi_proposals']} предложений для 99 снимков; {cover['spine_density_roi_proposals']} по плотности, два резервных контура без вымышленной межпозвоночной границы.",
            f"На десяти опубликованных DXA средний Dice ROI шейки: {n(roi['mean_dice_fixed'])} без адаптации и {n(roi['mean_dice_adapted'])} с исключением текущего DICOM-исследования из подбора шаблона.",
            'Это совпадение с измерительной ROI издателя, не точность полного контура кости или ориентиров в мм. Наличие предсказания не доказывает его анатомическую правильность.'])),
        ('Офлайн-проверка поставки', items([
            f"CLI и настоящий HTTP: 499/499 Success, ответы совпали, CSV/XLSX и 499 PNG + 499 SC + 499 SR проверены.",
            f"CLI: пакет {cli['total_seconds']:.1f} с; верхняя граница на исследование с инициализацией моделей {cold['cold_study_upper_bound_seconds']:.1f} с.",
            f"HTTP: пакет {http['batch_seconds']:.1f} с; верхняя граница на исследование {http['max_study_upper_bound_seconds']:.1f} с при лимите 180 с.",
            'Затем успешно обработан второй HTTP-пакет из трёх DICOM; очередь остаётся работоспособной. Сеть — только lo, runtime-загрузок весов нет.',
            'Локальная RTX 4070; H200 не измерялась. Загрузка Python и запуск Docker отдельно не входят в таймер инициализации CLI.'])),
        ('Что остаётся по ТЗ', items([
            'Наличие маски Th12 не измеряет видимую половину его тела. Точка большого вертела не доказывает исключение всей структуры из ROI.',
            'Область вне кости не является подтверждённой мягкой тканью. Разные виды ROI и охват сканирования требуют разных проверок.',
            'Прежний отчёт полноты смешивает вычисление проверки с клиническими флагами. Его нельзя исправить простым переключением флагов.',
            'Закрытый тест и экспертная приёмка остаются у организатора. Собственная клиническая сертификация не является условием данного ТЗ.',
            'requirements_complete остаётся false. Глобальный максимум метрик и качество закрытого теста не установлены.'])),
        ('Демонстрация и воспроизведение', items([
            './competition/local_run.sh serve 8091 — локальный GPU-сервис; ./competition/run.sh — отдельная CPU-поставка.',
            'Загрузить три отладочных DICOM, открыть координаты и ROI, показать предел 5°, источник масштаба, типы и неопределённости.',
            'Скачать таблицы, исходный ZIP и SC/SR; сравнить с CLI. Отладочный прогон не измеряет медицинскую точность.',
            'Код и агрегаты — в Git; большие веса, данные и переносимые Docker-архивы хранятся отдельно.',
            'Полные протоколы, SHA-256, сравнения и ограничения: docs/competition/anatomy_upgrade_2026_10_15.json.'])),
    ]
    old = args.output.read_text() if args.output.exists() else ''
    match = re.search(r'<style>(.*?)</style>', old, re.S)
    css = match.group(1) if match else 'body{font:20px/1.5 system-ui}main{max-width:1100px;margin:auto}.slide{padding:40px;break-after:page}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px}'
    body = ''.join(f'<section class="slide" aria-labelledby="s{i}"><p class="num">{i} / {len(sections)}</p><h1 id="s{i}">{escape(title)}</h1>{content}</section>' for i, (title, content) in enumerate(sections, 1))
    args.output.write_text(f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Osseo AI — результаты и ограничения 1.11.7</title><style>{css}</style></head><body><main>{body}</main></body></html>\n')
    print(str(args.output))


if __name__ == '__main__':
    main()
