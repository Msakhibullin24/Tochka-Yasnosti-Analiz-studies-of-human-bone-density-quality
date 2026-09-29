import { expect, it } from 'vitest'
import { buildDynamicsDescription, type DynamicsResponse } from './dynamicsDescription'

const result: DynamicsResponse = {
  verdict: 'review', verdict_ru: 'Сопоставимость требует проверки специалистом', anatomical_region: 'Поясничный отдел',
  checks: [{ id: 'quality', label: 'Качество', status: 'review', detail: 'изменена укладка' }],
  bmd_change: { baseline_bmd: 0.9, followup_bmd: 0.84, change_percent: -6.7, lsc_percent: 3,
    interpretation: 'Изменение значимо (подтвердить после проверки)' },
  tolerances_clinically_validated: false,
}

it('withholds significance when positioning needs review', () => {
  const text = buildDynamicsDescription(result)
  expect(text).toContain('Качество: требует проверки; изменена укладка')
  expect(text).toContain('-6.7%')
  expect(text).toContain('Значимость изменения не интерпретируется')
  expect(text).not.toContain('Изменение значимо')
})

it('reports unavailable LSC without inventing significance', () => {
  const text = buildDynamicsDescription({ ...result, verdict: 'comparable', bmd_change: { ...result.bmd_change!, lsc_percent: null } })
  expect(text).toContain('LSC не задан')
})
