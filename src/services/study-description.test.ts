import { describe, expect, it } from 'vitest'
import type { DatasetStudy } from './dataset'
import type { LongitudinalComparison, LongitudinalMeasurement } from './longitudinal-api'
import { buildStudyDescription } from './study-description'

const study = {
  studyId: 'ST-22222222222222222222', protocol: 'spine_pa',
  quality: { reviewRequired: true, flags: ['Контраст требует проверки'] },
  technicalQc: { score: 72, flags: ['Контраст требует проверки'] },
  annotations: [{
    evaluable: true, overallAction: 'review',
    defects: [{ code: 'spine_tilt', present: true, severity: 'major', comment: 'Проверить ось' },
      { code: 'vertebra_roi', present: true, severity: 'minor' }],
    landmarks: [{ name: 'L1', visible: true }], regions: [{ name: 'L1_roi' }],
    expert: { readerId: 'reader-01', comment: 'Контрольная отметка' },
  }],
} as unknown as DatasetStudy

const measurement = {
  studyId: study.studyId, acquiredOn: '2026-08-25', source: 'manual-verified',
  sourceReference: 'report-sha256:test', sites: [{ site: 'l1-l4', bmd: 0.891 }],
  qualityStatus: 'accept', positioningStatus: 'accept', roiStatus: 'review',
} as LongitudinalMeasurement

const comparison = {
  comparisonId: 'CMP-AAAAAAAAAAAAAAAAAAAA', currentStudyId: study.studyId,
  baselineDate: '2024-08-20', currentDate: '2026-08-25', intervalMonths: 24,
  status: 'review',
  sites: [{ site: 'l1-l4', baselineBmd: 0.842, currentBmd: 0.891,
    percentChange: 5.8, lscPercent: 5.3, status: 'not-comparable' }],
  checks: [{ id: 'roi', label: 'Сопоставимые ROI', status: 'review', detail: 'требует проверки' }],
} as LongitudinalComparison

describe('automatic study description', () => {
  it('includes each source finding and does not interpret blocked BMD change', () => {
    const text = buildStudyDescription(study, measurement, comparison, 'ru', {
      spine_tilt: { ru: 'Наклон позвоночника', en: 'Spine tilt' },
      vertebra_roi: { ru: 'ROI позвонка', en: 'Vertebra ROI' },
    })
    expect(text).toContain('Контраст требует проверки')
    expect(text).toContain('Наклон позвоночника (выраженная) — Проверить ось')
    expect(text).toContain('ROI позвонка (незначительная)')
    expect(text).toContain('L1_roi')
    expect(text).toContain('0.891 g/cm²')
    expect(text).toContain('+5.8%; LSC 5.3%; без интерпретации')
    expect(text).toContain('Сопоставимые ROI — требует проверки')
    expect(text).not.toContain('изменение выше LSC')
  })

  it('states when confirmed BMD and a prior comparison are absent', () => {
    const text = buildStudyDescription({ ...study, annotations: [] }, undefined, null, 'ru')
    expect(text).toContain('Экспертных отметок нет')
    expect(text).toContain('МПК не вычисляется')
    expect(text).toContain('Не рассчитана')
  })

  it('interprets change only after comparability passes', () => {
    const text = buildStudyDescription(study, measurement, {
      ...comparison, status: 'comparable', sites: [{ ...comparison.sites[0], status: 'significant-gain' }], checks: [],
    }, 'ru')
    expect(text).toContain('изменение выше LSC, прирост')
    expect(buildStudyDescription(study, measurement, {
      ...comparison, currentStudyId: 'ST-OTHER', status: 'comparable', checks: [],
    }, 'ru')).toContain('Не рассчитана')
  })

  it('keeps conflicting expert reads separate', () => {
    const second = { ...study.annotations![0], overallAction: 'accept' as const, defects: [] }
    const text = buildStudyDescription({ ...study, annotations: [study.annotations![0], second] }, undefined, null, 'ru')
    expect(text).toContain('Чтение 1')
    expect(text).toContain('Чтение 2')
    expect(text).toContain('Чтения расходятся')
  })
})
