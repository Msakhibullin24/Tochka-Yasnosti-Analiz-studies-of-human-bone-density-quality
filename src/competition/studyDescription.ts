import { modelVerdict, reviewLabels, reviewStatus, type Row } from './worklist'

type ExpertReview = {
  author?: string
  status?: string
  quality_class?: number | null
  violations?: string[]
  comment?: string
  geometry?: Array<{ name: string; kind: string; note?: string }>
  followups?: Array<{ kind: string; question: string; state: string; resolution?: string }>
}

const parse = <T,>(value: string | undefined, fallback: T): T => {
  try { return JSON.parse(value || '') as T } catch { return fallback }
}

export function buildQcStudyDescription(rows: Row[], criteria: Record<string, string> = {}): string {
  if (!rows.length) return 'В исследовании нет изображений.'
  const success = rows.filter((row) => row.processing_status === 'Success')
  const lines = [
    'Автоматическое описание исследования DXA',
    `Изображений: ${rows.length}; обработано: ${success.length}; ошибок обработки: ${rows.length - success.length}.`,
    'Текст составлен из ответов модели и статусов проверки. Он не является медицинским заключением.',
  ]
  rows.forEach((row, index) => {
    lines.push('', `Изображение ${index + 1}: ${row.path_to_file || 'путь не указан'}.`)
    if (row.processing_status !== 'Success') {
      lines.push(`Обработка не завершена: ${row.error_code || 'ошибка'}${row.error_message ? ` — ${row.error_message}` : ''}.`)
      return
    }
    lines.push(`Область: ${row.anatomical_region || 'не определена'}. ${modelVerdict(row)}.`)
    lines.push(`Экспертная проверка: ${reviewLabels[reviewStatus(row)] || 'статус неизвестен'}.`)
    const review = parse<ExpertReview | null>(row.review_document, null)
    if (review) {
      lines.push(`Экспертная версия ${row.review_revision || 'не указана'}: ${review.author || 'автор не указан'}; ${reviewLabels[review.status || ''] || review.status || 'статус неизвестен'}.`)
      if (review.quality_class !== null && review.quality_class !== undefined) {
        lines.push(`Решение специалиста: ${review.quality_class === 1 ? 'нарушение качества' : 'нарушений качества не отмечено'}.`)
      }
      review.violations?.forEach((code) => lines.push(`• Подтверждённое специалистом нарушение: ${criteria[code] || code}.`))
      if (review.comment) lines.push(`Комментарий специалиста: ${review.comment}`)
      review.geometry?.forEach((item) => lines.push(`• Экспертная разметка: ${item.name} (${item.kind})${item.note ? ` — ${item.note}` : ''}.`))
      review.followups?.forEach((action) => lines.push(`• Последующее действие (${action.state}): ${action.question}${action.resolution ? ` — ${action.resolution}` : ''}.`))
    }
    if (row.duplicate_of) lines.push('Повторный экспорт изображения; исходная ROI и результат проверки относятся к этому файлу.')
    const states = parse<Record<string, { status?: string }>>(row.criterion_states, {})
    for (const [code, state] of Object.entries(states)) {
      if (state.status === 'fail' || state.status === 'undetermined') {
        lines.push(`Критерий «${criteria[code] || code}»: ${state.status === 'fail' ? 'нарушение по модели' : 'не определён'}.`)
      }
    }
    const checks = parse<Array<{ id?: string; status?: string }>>(row.requirement_checks, [])
    const unresolved = checks.filter((check) => check.status === 'undetermined').map((check) => check.id).filter(Boolean)
    if (unresolved.length) lines.push(`Анатомические проверки не завершены: ${unresolved.join(', ')}.`)
    else if (row.anatomical_checks_complete === 'false') lines.push('Полная анатомическая проверка не подтверждена.')
    const projection = parse<{ value?: string; status?: string }>(row.projection_assessment, {})
    if (projection.value) lines.push(`Проекция: ${projection.value}; статус: ${projection.status || 'не указан'}.`)
    const roi = parse<{ status?: string }>(row.source_roi_assessment, {})
    if (roi.status) lines.push(`Исходная ROI: ${roi.status}.`)
  })
  lines.push('', 'Динамика МПК между визитами требует явного выбора двух исследований, подтверждённых значений МПК и LSC. По этим строкам она не вычислялась.')
  return lines.join('\n')
}
