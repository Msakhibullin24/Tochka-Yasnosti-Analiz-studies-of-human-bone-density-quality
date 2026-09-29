import { expect, it } from 'vitest'
import { buildQcStudyDescription } from './studyDescription'

it('includes all image findings and preserves failed and unresolved checks', () => {
  const text = buildQcStudyDescription([
    { path_to_file: 'a.dcm', processing_status: 'Success', anatomical_region: 'Позвоночник', quality_class: '1',
      violation_type: 'Не выравнена ось', criterion_states: '{"spine_axis":{"status":"fail"}}',
      requirement_checks: '[{"id":"Th12","status":"undetermined"}]', review_status: 'unreviewed',
      source_roi_assessment: '{"status":"absent"}' },
    { path_to_file: 'b.dcm', processing_status: 'Failure', error_code: 'UNSUPPORTED_PROJECTION', error_message: 'Боковая проекция' },
  ], { spine_axis: 'Ось позвоночника' })
  expect(text).toContain('Изображений: 2; обработано: 1; ошибок обработки: 1')
  expect(text).toContain('Критерий «Ось позвоночника»: нарушение по модели')
  expect(text).toContain('Анатомические проверки не завершены: Th12')
  expect(text).toContain('Исходная ROI: absent')
  expect(text).toContain('UNSUPPORTED_PROJECTION — Боковая проекция')
  expect(text).toContain('Динамика МПК')
})

it('includes the latest expert decision, markup and unresolved followups separately from model findings', () => {
  const text = buildQcStudyDescription([{ path_to_file: 'study/a.dcm', processing_status: 'Success',
    quality_class: '0', review_status: 'confirmed', review_revision: '2',
    review_document: JSON.stringify({ author: 'Врач', status: 'confirmed', quality_class: 1,
      violations: ['spine_axis'], comment: 'Ось смещена', geometry: [{ name: 'axis', kind: 'line', note: 'Уточнено' }],
      followups: [{ kind: 'second_opinion', question: 'Проверить Th12', state: 'open', resolution: '' }] }),
  }], { spine_axis: 'Наклон оси' })
  expect(text).toContain('Модель не выявила нарушений')
  expect(text).toContain('Экспертная версия 2: Врач; Подтверждено')
  expect(text).toContain('Подтверждённое специалистом нарушение: Наклон оси')
  expect(text).toContain('Ось смещена')
  expect(text).toContain('axis (line) — Уточнено')
  expect(text).toContain('Проверить Th12')
})
