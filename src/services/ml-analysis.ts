import type { LocalizedText, Study } from '../types'
import { analyzeDicom, DicomAnalysisError } from './analysis'

const API_BASE = (import.meta.env.VITE_ANALYSIS_API_URL || '/api/v1').replace(/\/$/, '')

const t = (ru: string, en: string): LocalizedText => ({ ru, en })

const isStudy = (value: unknown): value is Study => {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Partial<Study>
  return typeof candidate.id === 'string'
    && ['spine', 'hip', 'total-body'].includes(candidate.type ?? '')
    && ['passed', 'review', 'rejected'].includes(candidate.status ?? '')
    && Array.isArray(candidate.criteria)
    && Boolean(candidate.provenance)
}

const fallbackWithWarning = async (file: File, warning: LocalizedText) => {
  const study = await analyzeDicom(file)
  return {
    ...study,
    provenance: {
      ...study.provenance,
      warnings: [warning, ...study.provenance.warnings],
    },
  }
}

const errorCode = async (response: Response) => {
  try {
    const body = await response.json() as { detail?: { code?: string } }
    return body.detail?.code
  } catch {
    return undefined
  }
}

export async function analyzeStudy(file: File): Promise<Study> {
  if (typeof fetch !== 'function') return analyzeDicom(file)
  const formData = new FormData()
  formData.append('file', file)
  let response: Response
  try {
    response = await fetch(`${API_BASE}/studies/analyze`, { method: 'POST', body: formData })
  } catch {
    return fallbackWithWarning(file, t(
      'ML-сервис недоступен; выполнен локальный технический pre-screening.',
      'The ML service is unavailable; local technical pre-screening was used.',
    ))
  }
  if (response.ok) {
    const study: unknown = await response.json()
    if (!isStudy(study)) throw new DicomAnalysisError('invalid-dicom', 'The inference service returned an invalid study response.')
    return study
  }
  const code = await errorCode(response)
  if (response.status === 422 && code === 'UNSUPPORTED_PROTOCOL') {
    return fallbackWithWarning(file, t(
      'Total-body модель не применяется к этому протоколу; выполнен технический pre-screening.',
      'The total-body model does not support this protocol; technical pre-screening was used.',
    ))
  }
  if ([404, 502, 503, 504].includes(response.status)) {
    return fallbackWithWarning(file, t(
      'Исследовательская модель не готова к inference; выполнен локальный технический pre-screening.',
      'The research model is not ready for inference; local technical pre-screening was used.',
    ))
  }
  if (response.status === 413) throw new DicomAnalysisError('file-too-large', 'The DICOM file exceeds the size limit.')
  if (code === 'EMPTY_FILE') throw new DicomAnalysisError('empty-file', 'The DICOM file is empty.')
  throw new DicomAnalysisError('invalid-dicom', 'The inference service could not process this DICOM file.')
}
