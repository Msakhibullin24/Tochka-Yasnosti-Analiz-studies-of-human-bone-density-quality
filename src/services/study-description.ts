import type { DatasetStudy } from './dataset'
import type { LongitudinalComparison, LongitudinalMeasurement, MeasurementSite } from './longitudinal-api'
import type { Locale } from '../types'

type Labels = Record<string, { ru: string; en: string }>
const siteNames: Record<MeasurementSite, { ru: string; en: string }> = {
  'l1-l4': { ru: 'L1–L4', en: 'L1–L4' },
  'total-hip': { ru: 'Всё бедро', en: 'Total hip' },
  'femoral-neck': { ru: 'Шейка бедра', en: 'Femoral neck' },
}
const protocolNames: Record<DatasetStudy['protocol'], { ru: string; en: string }> = {
  spine_pa: { ru: 'Поясничный отдел', en: 'Lumbar spine' },
  hip_left: { ru: 'Левое бедро', en: 'Left hip' },
  hip_right: { ru: 'Правое бедро', en: 'Right hip' },
  forearm_left: { ru: 'Левое предплечье', en: 'Left forearm' },
  forearm_right: { ru: 'Правое предплечье', en: 'Right forearm' },
  total_body: { ru: 'Всё тело', en: 'Total body' },
  unsupported: { ru: 'Протокол не определён', en: 'Protocol unknown' },
}
const actionNames = {
  accept: { ru: 'принять', en: 'accept' },
  review: { ru: 'проверить', en: 'review' },
  repeat: { ru: 'повторить', en: 'repeat' },
}
const statusNames = {
  accept: { ru: 'принято', en: 'accepted' },
  review: { ru: 'требует проверки', en: 'review required' },
  reject: { ru: 'отклонено', en: 'rejected' },
}
const severityNames = {
  none: { ru: 'не указана', en: 'unspecified' },
  minor: { ru: 'незначительная', en: 'minor' },
  major: { ru: 'выраженная', en: 'major' },
  critical: { ru: 'критическая', en: 'critical' },
}

export function buildStudyDescription(
  study: DatasetStudy,
  measurement: LongitudinalMeasurement | undefined,
  comparison: LongitudinalComparison | null,
  locale: Locale,
  defectLabels: Labels = {},
): string {
  const ru = locale === 'ru'
  const lines: string[] = [
    ru ? 'Автоматическое описание исследования' : 'Automatic study description',
    `${ru ? 'Исследование' : 'Study'}: ${study.studyId}. ${ru ? 'Область' : 'Region'}: ${protocolNames[study.protocol][locale]}.`,
    ru
      ? 'Черновик по доступным данным. Технический скрининг и экспертные отметки имеют разные источники; текст не является медицинским заключением.'
      : 'Draft from available data. Technical screening and expert annotations have different sources; this is not a clinical report.',
    '',
    ru ? 'Техническая проверка' : 'Technical screening',
  ]
  const flags = [...new Set([...(study.technicalQc?.flags ?? []), ...(study.quality?.flags ?? [])])]
  if (study.technicalQc) lines.push(`${ru ? 'Балл технического baseline' : 'Technical baseline score'}: ${study.technicalQc.score}/100.`)
  if (flags.length) flags.forEach((flag) => lines.push(`• ${flag}`))
  else lines.push(ru ? 'Технические флаги не выявлены; анатомия и ROI этим не подтверждены.' : 'No technical flags detected; anatomy and ROI are not confirmed by this check.')
  if (study.quality?.reviewRequired) lines.push(ru ? 'Исследование требует экспертной проверки.' : 'This study requires expert review.')
  study.technicalQc?.limitations?.forEach((limit) => lines.push(`${ru ? 'Ограничение baseline' : 'Baseline limitation'}: ${limit}`))

  lines.push('', ru ? 'Экспертные отметки' : 'Expert annotations')
  const annotations = study.annotations ?? []
  if (!annotations.length) lines.push(ru ? 'Экспертных отметок нет.' : 'No expert annotations are available.')
  annotations.forEach((read, index) => {
    lines.push(`${ru ? 'Чтение' : 'Read'} ${index + 1}: ${read.expert.readerId}; ${ru ? 'действие' : 'action'} — ${actionNames[read.overallAction][locale]}; ${ru ? 'оценимость' : 'evaluable'} — ${read.evaluable ? (ru ? 'да' : 'yes') : (ru ? 'нет' : 'no')}.`)
    if (!read.evaluable && read.notEvaluableReason) lines.push(`${ru ? 'Причина' : 'Reason'}: ${read.notEvaluableReason}`)
    const found = (read.defects ?? []).filter((defect) => defect.present)
    if (found.length) found.forEach((defect) => {
      const name = defectLabels[defect.code]?.[locale] ?? defect.code
      lines.push(`• ${name} (${severityNames[defect.severity][locale]})${defect.comment ? ` — ${defect.comment}` : ''}`)
    })
    else lines.push(ru ? 'Отмеченных дефектов нет.' : 'No defects marked.')
    const landmarks = (read.landmarks ?? []).filter((landmark) => landmark.visible).map((landmark) => landmark.name)
    if (landmarks.length) lines.push(`${ru ? 'Отмеченные ориентиры' : 'Marked landmarks'}: ${landmarks.join(', ')}.`)
    if (read.regions?.length) lines.push(`${ru ? 'Отмеченные области' : 'Marked regions'}: ${read.regions.map((region) => region.name).join(', ')}.`)
    if (read.expert.comment) lines.push(`${ru ? 'Комментарий' : 'Comment'}: ${read.expert.comment}`)
  })
  if (annotations.length > 1) {
    const conclusions = annotations.map((read) => `${read.overallAction}:${read.defects.filter((defect) => defect.present).map((defect) => defect.code).sort().join(',')}`)
    if (new Set(conclusions).size > 1) lines.push(ru
      ? 'Чтения расходятся по действию или дефектам; единое экспертное заключение автоматически не составляется.'
      : 'Reads disagree on action or defects; no unified expert conclusion is generated automatically.')
  }

  lines.push('', ru ? 'Подтверждённая МПК' : 'Confirmed BMD')
  if (measurement) {
    lines.push(`${ru ? 'Дата' : 'Date'}: ${measurement.acquiredOn}. ${ru ? 'Источник' : 'Source'}: ${measurement.source}; ${measurement.sourceReference}.`)
    measurement.sites.forEach((site) => lines.push(`• ${siteNames[site.site][locale]}: ${site.bmd.toFixed(3)} g/cm².`))
    lines.push(`${ru ? 'Качество' : 'Quality'}: ${statusNames[measurement.qualityStatus][locale]}; ${ru ? 'укладка' : 'positioning'}: ${statusNames[measurement.positioningStatus][locale]}; ROI: ${statusNames[measurement.roiStatus][locale]}.`)
  } else lines.push(ru ? 'Подтверждённое измерение МПК не внесено; по изображению МПК не вычисляется.' : 'No confirmed BMD measurement is available; BMD is not derived from the image.')

  lines.push('', ru ? 'Динамика' : 'Longitudinal change')
  if (!comparison || comparison.currentStudyId !== study.studyId) {
    lines.push(ru ? 'Не рассчитана. Выберите предыдущее исследование того же пациента и действующий профиль LSC.' : 'Not calculated. Select a prior study of the same patient and an active LSC profile.')
  } else {
    lines.push(`${ru ? 'Сравнение' : 'Comparison'}: ${comparison.comparisonId}; ${comparison.baselineDate} → ${comparison.currentDate}; ${comparison.intervalMonths} ${ru ? 'мес.' : 'months'}.`)
    const comparable = comparison.status === 'comparable'
    lines.push(comparable ? (ru ? 'Проверки сопоставимости пройдены.' : 'Comparability checks passed.') :
      (ru ? 'Динамика МПК не интерпретируется до разрешения замечаний по сопоставимости.' : 'BMD change is not interpreted until comparability issues are resolved.'))
    comparison.sites.forEach((site) => {
      const value = `${site.baselineBmd.toFixed(3)} → ${site.currentBmd.toFixed(3)} g/cm²; ${site.percentChange > 0 ? '+' : ''}${site.percentChange.toFixed(1)}%; LSC ${site.lscPercent.toFixed(1)}%`
      const interpretation = !comparable ? (ru ? 'без интерпретации' : 'not interpreted') :
        site.status === 'significant-gain' ? (ru ? 'изменение выше LSC, прирост' : 'change exceeds LSC, gain') :
        site.status === 'significant-loss' ? (ru ? 'изменение выше LSC, снижение' : 'change exceeds LSC, loss') :
        (ru ? 'изменение не превышает LSC' : 'change does not exceed LSC')
      lines.push(`• ${siteNames[site.site][locale]}: ${value}; ${interpretation}.`)
    })
    comparison.checks.filter((check) => check.status !== 'pass').forEach((check) =>
      lines.push(`• ${ru ? 'Ограничение' : 'Limitation'}: ${check.label} — ${check.detail}`))
  }
  lines.push('', ru ? 'Описание нужно проверить по исходным изображениям и документам измерения.' : 'Review this description against the source images and measurement records.')
  return lines.join('\n')
}
