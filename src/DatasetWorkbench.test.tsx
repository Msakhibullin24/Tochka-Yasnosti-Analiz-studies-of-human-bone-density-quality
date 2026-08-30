import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import DatasetWorkbench from './DatasetWorkbench'
import type { DatasetStudy, DatasetSummary } from './services/dataset'

const study: DatasetStudy = {
  schemaVersion: '1.0.0',
  loaderVersion: 'hologic-apex-loader/1.0.0',
  studyId: 'ST-0123456789ABCDEFFEDC',
  patientGroupId: 'PG-0123456789ABCDEFFEDC',
  deviceGroup: 'DV-0123456789ABCDEFFEDC',
  protocol: 'spine_pa',
  protocolCode: 'AP Spine',
  acquisitionYear: '2026',
  softwareVersion: '13.3.0.1',
  split: 'train',
  processedImages: [{ tag: '0x0046', width: 400, height: 600, p01: 1, p99: 254, clippedFraction: 0 }],
  raw: { shape: [600, 400, 6], transmissionCount: 6, phaseSemantics: 'unverified', saturatedFraction: 0, phases: [] },
  quality: { reviewRequired: false, flags: [] },
  technicalQc: { baselineVersion: 'osseo-technical-qc/1.0.0', score: 100, trainingCandidate: true, requiresExpertReview: false, flags: [], processed: [], raw: { saturatedFraction: 0, phaseSemantics: 'unverified' }, limitations: [] },
  assets: [{ name: 'p_0046.png', tag: '0x0046', url: `/api/v1/datasets/current/studies/ST-0123456789ABCDEFFEDC/assets/p_0046.png` }],
  rawChannels: Array.from({ length: 6 }, (_, index) => ({ index, label: `Канал ${index}`, url: `/api/v1/datasets/current/studies/ST-0123456789ABCDEFFEDC/raw/${index}.png` })),
  annotationCount: 0,
  annotations: [],
}

const summary: DatasetSummary = {
  schemaVersion: '1.0.0',
  studyCount: 1,
  patientGroupCount: 1,
  annotatedStudyCount: 0,
  annotationCount: 0,
  reviewRequiredCount: 0,
  protocols: { spine_pa: 1 },
  splits: { train: 1 },
  phaseSemantics: 'unverified',
}

const readiness = {
  schemaVersion: '1.0.0',
  policyVersion: 'ru-dxa-qc/1.0.0',
  intendedUse: { ru: 'Исследовательская система; не ставит диагноз.', en: 'Research decision support; no diagnosis.' },
  datasetVersion: 'DS-0123456789ABCDEF',
  stage: 'research',
  criticalBlockerCount: 4,
  warningCount: 1,
  gates: [{ id: 'dataset_integrity', label: 'Целостность датасета', status: 'pass', critical: true, evidence: 'blockers=0', nextAction: '' }],
  integrity: { status: 'pass', studyCount: 1, patientGroupCount: 1, processedAssetCount: 1, blockerCount: 0, warningCount: 0 },
  agreement: { annotationCoverage: 0, doubleReadCoverage: 0, pendingAdjudicationCount: 0, overallAction: { comparisonCount: 0, percentAgreement: null, cohenKappa: null } },
  audit: { valid: true, eventCount: 0, headHash: '0'.repeat(64) },
}

afterEach(() => vi.unstubAllGlobals())

describe('Dataset Workbench', () => {
  it('loads a de-identified study and saves an expert annotation', async () => {
    const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input)
      if (init?.method === 'PUT') return { ok: true, status: 200, json: async () => JSON.parse(String(init.body)) }
      if (url.endsWith('/exclusions')) return { ok: true, status: 200, json: async () => ({ exclusions: [], count: 4 }) }
      if (url.endsWith('/readiness')) return { ok: true, status: 200, json: async () => readiness }
      if (url.includes('/longitudinal/patients/')) return { ok: true, status: 200, json: async () => ({ patientGroupId: study.patientGroupId, measurements: [], count: 0 }) }
      if (url.endsWith('/longitudinal/lsc-profiles')) return { ok: true, status: 200, json: async () => ({ profiles: [], count: 0 }) }
      if (url.endsWith('/longitudinal/cross-calibrations')) return { ok: true, status: 200, json: async () => ({ calibrations: [], count: 0 }) }
      if (url.endsWith('/datasets/current/studies')) return { ok: true, status: 200, json: async () => ({ studies: [study], count: 1 }) }
      if (url.endsWith(study.studyId)) return { ok: true, status: 200, json: async () => study }
      return { ok: true, status: 200, json: async () => summary }
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<DatasetWorkbench locale="ru" />)
    expect(await screen.findByRole('heading', { name: 'Контур данных и разметки' })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Экспертная разметка' })).toBeInTheDocument()
    expect(screen.getByText('Secondary Capture · trainingEligible=false')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Готовность к клиническому пилоту' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Реальный анализ в динамике' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /COCO/ })).toHaveAttribute('href', '/api/v1/datasets/current/exports/coco')

    fireEvent.change(screen.getByLabelText('ID эксперта'), { target: { value: 'reader-01' } })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining(`/studies/${study.studyId}/annotations`),
      expect.objectContaining({ method: 'PUT' }),
    ))
    expect(await screen.findByText('Разметка сохранена')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('BMD L1–L4, g/cm²'), { target: { value: '0.842' } })
    fireEvent.change(screen.getByLabelText('Ссылка на источник'), { target: { value: 'report-sha256:test' } })
    fireEvent.change(screen.getByLabelText('ID подтвердившего эксперта'), { target: { value: 'reader-01' } })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить измерение' }))
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining(`/longitudinal/measurements/${study.studyId}`),
      expect.objectContaining({ method: 'PUT' }),
    ))
    expect(await screen.findByText('Измерение сохранено и добавлено во временной ряд.')).toBeInTheDocument()
  })

  it('shows an actionable offline state when no dataset is configured', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ detail: { code: 'DATASET_UNAVAILABLE', message: 'OSSEO_DATASET_ROOT is not configured' } }),
    }))
    render(<DatasetWorkbench locale="ru" />)
    expect(await screen.findByRole('heading', { name: 'Обезличенный dataset не подключён' })).toBeInTheDocument()
    expect(screen.getByText(/OSSEO_PSEUDONYM_KEY/)).toBeInTheDocument()
  })
})
