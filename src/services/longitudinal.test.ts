import { describe, expect, it } from 'vitest'
import { calculateSiteChange, classifyBmdChange, monthsBetween, validateTrendHistory } from './longitudinal'

describe('longitudinal BMD analysis', () => {
  it('classifies only changes reaching the facility LSC as significant', () => {
    expect(classifyBmdChange(5.3, 5.3, true)).toBe('significant-gain')
    expect(classifyBmdChange(5.2, 5.3, true)).toBe('stable')
    expect(classifyBmdChange(-5.3, 5.3, true)).toBe('significant-loss')
  })

  it('never interprets change when studies are not comparable', () => {
    expect(classifyBmdChange(-12, 5.3, false)).toBe('not-comparable')
  })

  it('calculates absolute and percentage change with clinical rounding', () => {
    const result = calculateSiteChange({
      site: 'l1-l4',
      label: { ru: 'L1–L4', en: 'L1–L4' },
      baselineBmd: 0.842,
      currentBmd: 0.891,
      lscPercent: 5.3,
      history: [
        { studyId: 'baseline', date: '2024-08-20', bmd: 0.842 },
        { studyId: 'current', date: '2026-08-25', bmd: 0.891 },
      ],
    }, 'comparable')

    expect(result.absoluteChange).toBe(0.049)
    expect(result.percentChange).toBe(5.8)
    expect(result.status).toBe('significant-gain')
  })

  it('calculates calendar month intervals and validates chronological history', () => {
    expect(monthsBetween('2024-08-20', '2026-08-25')).toBe(24)
    expect(validateTrendHistory([
      { studyId: 'a', date: '2024-08-20', bmd: 0.842 },
      { studyId: 'b', date: '2026-08-25', bmd: 0.891 },
    ])).toBe(true)
    expect(validateTrendHistory([
      { studyId: 'b', date: '2026-08-25', bmd: 0.891 },
      { studyId: 'a', date: '2024-08-20', bmd: 0.842 },
    ])).toBe(false)
  })

  it('rejects invalid BMD, LSC, and time-series inputs', () => {
    expect(() => calculateSiteChange({
      site: 'l1-l4',
      label: { ru: 'L1–L4', en: 'L1–L4' },
      baselineBmd: 0,
      currentBmd: 0.891,
      lscPercent: 5.3,
      history: [{ studyId: 'only', date: '2026-08-25', bmd: 0.891 }],
    }, 'comparable')).toThrow('Invalid longitudinal BMD input')
  })
})
