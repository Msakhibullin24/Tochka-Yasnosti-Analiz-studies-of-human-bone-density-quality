"""Shared wording for machine results and the limits of their interpretation."""
import json

REVIEW_REASONS = {
    'axis_measurement_unavailable': 'Угол оси недоступен для оценки.',
    'axis_model_measurement_disagreement': 'Измеренный угол и классификатор оси расходятся; проверьте измерение.',
    'violation_type_undetermined': 'Бинарный детектор выявил нарушение, но тип не установлен.',
    'suspected_metal_requires_review': 'Яркий участок требует проверки на металл; ошибка ROI автоматически не установлена.',
    'hip_geometry_unstable': 'Ориентиры или измерения бедра меняются при технической проверке изображения; проверьте их вручную.',
}


def parsed(row, key, default):
    value = row.get(key)
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value) if value else default
    except (ValueError, TypeError):
        return default


def model_verdict(row):
    if str(row.get('quality_class')) == '1':
        return row.get('violation_type') or 'Модель выявила нарушение; тип не установлен'
    return 'Модель не выявила нарушений; полнота анатомической проверки отдельно'


def assessment_notes(row):
    notes = ['Полная анатомическая проверка не подтверждена. Результат модели не является заключением о пригодности исследования.']
    projection = parsed(row, 'projection_assessment', {})
    if projection:
        notes.append('Проекция: ' + ('гипотеза фронтальной, не валидирована.' if projection.get('value') == 'frontal' else 'не определена по изображению.'))
    anatomy = parsed(row, 'anatomy_assessment', {})
    missing = [x['name'] for x in anatomy.get('landmarks', []) if not x.get('points')]
    if missing:
        notes.append('Не локализованы ориентиры: ' + ', '.join(missing) + '.')
    roi = parsed(row, 'source_roi_assessment', {})
    if roi:
        notes.append('Исходная ROI: ' + {'absent': 'отсутствует в поддерживаемых структурах.',
                     'unavailable': 'не прочитана.', 'partial': 'прочитана частично; анатомическая корректность не подтверждена.',
                     'extracted': 'прочитана; анатомическая корректность не подтверждена.'}.get(roi.get('status'), 'состояние неизвестно.'))
    for reason in parsed(row, 'review_reasons', []):
        notes.append(REVIEW_REASONS.get(reason, reason))
    return notes
