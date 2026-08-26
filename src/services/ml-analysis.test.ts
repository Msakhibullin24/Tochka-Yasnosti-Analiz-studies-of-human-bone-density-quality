import { afterEach, describe, expect, it, vi } from 'vitest'
import { analyzeStudy } from './ml-analysis'

const backendStudy = {
  id: 'ST-A1B2C3D4', patientId: 'P-11223344', filename: 'DICOM-A1B2C3D4.dcm', acquiredAt: '—',
  type: 'total-body', status: 'review', score: 79, confidence: 91, device: 'Hologic Discovery',
  institution: { ru: 'Скрыто', en: 'Masked' }, operator: 'MASKED', accessionNumber: 'ACC-55667788', seriesUid: 'UID-99AABBCC',
  technical: { modality: 'DX', rows: 1914, columns: 654 },
  provenance: { mode: 'research-model', modelVersion: 'hawaii-ai-dxa-points-7ac19eb', criteriaVersion: 'DXA-TOTAL-BODY-QC-2026.1', processedAt: '2026-08-26T00:00:00Z', warnings: [] },
  privacy: { deidentificationVerified: true, burnedInAnnotation: 'NO' },
  landmarks: [{ name: 'crown', x: .5, y: .02, confidence: .98, visible: true }],
  criteria: [{ id: 'landmarks', title: { ru: 'Ориентиры', en: 'Landmarks' }, detail: { ru: 'Определены', en: 'Located' }, status: 'pass', confidence: 98 }],
  recommendation: { ru: 'Проверить', en: 'Review' },
}

afterEach(() => vi.unstubAllGlobals())

describe('ML analysis adapter', () => {
  it('maps a successful total-body backend result into the Study contract', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => backendStudy })
    vi.stubGlobal('fetch', fetchMock)
    const result = await analyzeStudy(new File(['DICOM'], 'whole-body.dcm', { type: 'application/dicom' }))
    expect(result.type).toBe('total-body')
    expect(result.provenance.mode).toBe('research-model')
    expect(result.landmarks).toHaveLength(1)
    expect(fetchMock).toHaveBeenCalledWith(expect.stringMatching(/\/studies\/analyze$/), expect.objectContaining({ method: 'POST' }))
  })
})
