export type DynamicsResponse = {
  verdict: 'comparable' | 'review' | 'not_comparable'
  verdict_ru: string
  anatomical_region: string
  checks: Array<{ id: string; label: string; status: 'pass' | 'review' | 'block'; detail: string; baseline?: number | null; followup?: number | null; delta?: number | null; unit?: string }>
  bmd_change: { baseline_bmd: number; followup_bmd: number; change_percent: number; lsc_percent: number | null; interpretation: string } | null
  tolerances_clinically_validated: boolean
}

export function buildDynamicsDescription(value: DynamicsResponse): string {
  const lines = [
    'Автоматическое описание сравнения DXA',
    `Область: ${value.anatomical_region}. ${value.verdict_ru}.`,
    'Сопоставление пациента и дат подтверждено пользователем; алгоритм не проверяет эти данные самостоятельно.',
    `Пороги сопоставимости клинически ${value.tolerances_clinically_validated ? 'валидированы' : 'не валидированы'}.`,
    '', 'Проверки:',
  ]
  const checkStatuses = { pass: 'пройдено', review: 'требует проверки', block: 'блокирует сравнение' }
  value.checks.forEach((check) => lines.push(`• ${check.label}: ${checkStatuses[check.status]}; ${check.detail}.`))
  lines.push('', 'Изменение МПК:')
  if (!value.bmd_change) lines.push('Подтверждённые значения МПК не введены; изменение не рассчитывалось.')
  else {
    const bmd = value.bmd_change
    lines.push(`${bmd.baseline_bmd.toFixed(3)} → ${bmd.followup_bmd.toFixed(3)} g/cm²; ${bmd.change_percent > 0 ? '+' : ''}${bmd.change_percent.toFixed(1)}%.`)
    if (value.verdict !== 'comparable') lines.push('Значимость изменения не интерпретируется: сопоставимость снимков не подтверждена.')
    else if (bmd.lsc_percent === null) lines.push('LSC не задан; значимость изменения не установлена.')
    else lines.push(`LSC: ${bmd.lsc_percent.toFixed(1)}%. ${bmd.interpretation}.`)
  }
  lines.push('', 'Описание является исследовательским черновиком и требует проверки исходных снимков и документов измерения.')
  return lines.join('\n')
}
