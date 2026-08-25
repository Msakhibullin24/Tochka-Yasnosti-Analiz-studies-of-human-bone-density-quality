import type { ComparisonStatus, SiteChange, TrendPoint, TrendStatus } from '../types'

const round = (value: number, digits: number) => Number(value.toFixed(digits))

export function classifyBmdChange(percentChange: number, lscPercent: number, comparable: boolean): TrendStatus {
  if (!comparable) return 'not-comparable'
  if (percentChange >= lscPercent) return 'significant-gain'
  if (percentChange <= -lscPercent) return 'significant-loss'
  return 'stable'
}

export function calculateSiteChange(
  input: Omit<SiteChange, 'absoluteChange' | 'percentChange' | 'status'>,
  comparisonStatus: ComparisonStatus,
): SiteChange {
  if (![input.baselineBmd, input.currentBmd, input.lscPercent].every(Number.isFinite)
    || input.baselineBmd <= 0 || input.currentBmd <= 0 || input.lscPercent <= 0
    || !validateTrendHistory(input.history)) {
    throw new Error('Invalid longitudinal BMD input')
  }
  const rawChange = input.currentBmd - input.baselineBmd
  const absoluteChange = round(rawChange, 3)
  const percentChange = round((rawChange / input.baselineBmd) * 100, 1)
  return {
    ...input,
    absoluteChange,
    percentChange,
    status: classifyBmdChange(percentChange, input.lscPercent, comparisonStatus === 'comparable'),
  }
}

export function monthsBetween(from: string, to: string) {
  const start = new Date(`${from}T00:00:00Z`)
  const end = new Date(`${to}T00:00:00Z`)
  return Math.max(0, (end.getUTCFullYear() - start.getUTCFullYear()) * 12 + end.getUTCMonth() - start.getUTCMonth())
}

export function validateTrendHistory(points: TrendPoint[]) {
  return points.length >= 2
    && points.every((point) => Number.isFinite(point.bmd) && point.bmd > 0 && /^\d{4}-\d{2}-\d{2}$/.test(point.date))
    && points.every((point, index) => index === 0 || point.date > points[index - 1].date)
}
