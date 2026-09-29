import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import DatasetLongitudinal from './DatasetLongitudinal'
import type { DatasetStudy } from './services/dataset'
import type { LongitudinalMeasurement, LscProfile } from './services/longitudinal-api'

const currentStudy = {
  studyId: 'ST-22222222222222222222', patientGroupId: 'PG-AAAAAAAAAAAAAAAAAAAA',
  deviceGroup: 'DV-AAAAAAAAAAAAAAAAAAAA', protocol: 'spine_pa',
} as DatasetStudy
const priorStudy = { ...currentStudy, studyId: 'ST-11111111111111111111' } as DatasetStudy

const measurement = (studyId: string, acquiredOn: string, bmd: number): LongitudinalMeasurement => ({
  schemaVersion: '2.0.0', studyId, patientGroupId: currentStudy.patientGroupId, acquiredOn,
  protocol: 'spine_pa', deviceGroup: currentStudy.deviceGroup, facilityId: 'FACILITY-01',
  operatorGroup: 'OPERATORS-01', qualityStatus: 'accept', positioningStatus: 'accept', roiStatus: 'accept',
  source: 'manual-verified', sourceReference: 'report-sha256:test', expertConfirmed: true,
  confirmedBy: 'ACT-TEST', sites: [{ site: 'l1-l4', bmd }],
})

const profile: LscProfile = {
  schemaVersion: '2.0.0', profileId: 'LSC-TEST-SPINE', version: '1.0.0', facilityId: 'FACILITY-01',
  deviceGroup: currentStudy.deviceGroup, operatorGroup: 'OPERATORS-01', protocol: 'spine_pa',
  validFrom: '2024-01-01', validTo: '2027-12-31', sites: [{ site: 'l1-l4', percent: 5.3 }],
  precisionStudyReference: 'precision-test', approvedBy: 'ACT-PHYSICIST', status: 'active',
}

afterEach(() => vi.unstubAllGlobals())

describe('structured longitudinal workflow', () => {
  it('selects a prior measurement and renders a validated LSC comparison', async () => {
    const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith('/longitudinal/lsc-profiles')) return { ok: true, json: async () => ({ profiles: [profile], count: 1 }) }
      if (url.endsWith('/longitudinal/cross-calibrations')) return { ok: true, json: async () => ({ calibrations: [], count: 0 }) }
      if (url.includes('/longitudinal/patients/')) return { ok: true, json: async () => ({ patientGroupId: currentStudy.patientGroupId, measurements: [measurement(priorStudy.studyId, '2024-08-20', 0.842), measurement(currentStudy.studyId, '2026-08-25', 0.891)], count: 2 }) }
      if (url.endsWith('/longitudinal/compare') && init?.method === 'POST') return { ok: true, json: async () => ({
        schemaVersion: '2.0.0', comparisonId: 'CMP-AAAAAAAAAAAAAAAAAAAA', engineVersion: 'osseo-longitudinal/2.0.0', decisionBasis: 'deterministic-quality-gates',
        baselineStudyId: priorStudy.studyId, currentStudyId: currentStudy.studyId, patientGroupId: currentStudy.patientGroupId,
        baselineDate: '2024-08-20', currentDate: '2026-08-25', intervalMonths: 24, lscProfileId: profile.profileId,
        lscProfileVersion: '1.0.0', crossCalibrationId: null, status: 'comparable', assumed: false,
        checks: [{ id: 'patient', label: 'Один пациент', status: 'pass', passed: true, critical: true, detail: 'patientGroupId совпадает' }],
        sites: [{ site: 'l1-l4', baselineBmd: 0.842, currentBmd: 0.891, absoluteChange: 0.049, percentChange: 5.8, lscPercent: 5.3, status: 'significant-gain' }],
        recommendationCode: 'INTERPRET_WITH_CLINICAL_CONTEXT', comparedAt: '2026-08-30T10:00:00Z',
      }) }
      return { ok: false, status: 404, json: async () => ({ detail: { code: 'NOT_FOUND', message: url } }) }
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<DatasetLongitudinal study={currentStudy} studies={[currentStudy, priorStudy]} locale="ru" />)

    expect(screen.getByRole('heading', { name: 'Автоматическое описание исследования' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Скачать описание' })).toBeInTheDocument()
    expect(await screen.findByText('Временной ряд пациента')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Baseline'), { target: { value: priorStudy.studyId } })
    fireEvent.change(screen.getByLabelText('Профиль LSC учреждения'), { target: { value: profile.profileId } })
    fireEvent.click(screen.getByRole('button', { name: 'Проверить и рассчитать' }))
    expect(await screen.findByRole('heading', { name: 'Сравнение допустимо' })).toBeInTheDocument()
    expect(screen.getByText('+5.8%')).toBeInTheDocument()
    expect(screen.getByText(/Проверки сопоставимости пройдены/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Baseline'), { target: { value: '' } })
    expect(screen.queryByRole('heading', { name: 'Сравнение допустимо' })).not.toBeInTheDocument()
    expect(screen.getByText(/Не рассчитана. Выберите предыдущее исследование/)).toBeInTheDocument()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/longitudinal/compare'), expect.objectContaining({ method: 'POST' })))
  })
})
