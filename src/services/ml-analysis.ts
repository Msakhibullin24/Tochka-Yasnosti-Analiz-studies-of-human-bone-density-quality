import type { Criterion, LocalizedText, Study } from '../types'
import { analyzeDicom, DicomAnalysisError } from './analysis'

const API_BASE = (import.meta.env.VITE_ANALYSIS_API_URL || '/api/v1').replace(/\/$/, '')
const REQUEST_TIMEOUT_MS = 30_000

const t = (ru: string, en: string): LocalizedText => ({ ru, en })

const isLocalizedText = (value: unknown): value is LocalizedText => Boolean(value)
  && typeof value === 'object'
  && typeof (value as LocalizedText).ru === 'string'
  && typeof (value as LocalizedText).en === 'string'

const isCriterion = (value: unknown): value is Criterion => Boolean(value)
  && typeof value === 'object'
  && typeof (value as Criterion).id === 'string'
  && isLocalizedText((value as Criterion).title)
  && isLocalizedText((value as Criterion).detail)
  && ['pass', 'warning', 'fail'].includes((value as Criterion).status)
  && Number.isFinite((value as Criterion).confidence)

const isStudy = (value: unknown): value is Study => {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Partial<Study>
  return typeof candidate.id === 'string'
    && typeof candidate.patientId === 'string'
    && typeof candidate.filename === 'string'
    && ['spine', 'hip', 'total-body'].includes(candidate.type ?? '')
    && ['passed', 'review', 'rejected'].includes(candidate.status ?? '')
    && Number.isFinite(candidate.score)
    && Number.isFinite(candidate.confidence)
    && Array.isArray(candidate.criteria) && candidate.criteria.every(isCriterion)
    && Boolean(candidate.technical && typeof candidate.technical === 'object')
    && Boolean(candidate.privacy && typeof candidate.privacy.deidentificationVerified === 'boolean')
    && Boolean(candidate.provenance && typeof candidate.provenance.modelVersion === 'string' && Array.isArray(candidate.provenance.warnings))
    && isLocalizedText(candidate.recommendation)
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
  const controller = new AbortController()
  const timeout = globalThis.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try {
    response = await fetch(`${API_BASE}/studies/analyze`, { method: 'POST', body: formData, signal: controller.signal })
  } catch {
    return fallbackWithWarning(file, t(
      'ML-сервис недоступен; выполнен локальный технический pre-screening.',
      'The ML service is unavailable; local technical pre-screening was used.',
    ))
  } finally {
    globalThis.clearTimeout(timeout)
  }
  if (response.ok) {
    try {
      const study: unknown = await response.json()
      if (isStudy(study)) return study
    } catch {
      // A valid DICOM should still receive a local result when the service contract breaks.
    }
    return fallbackWithWarning(file, t(
      'ML-сервис вернул некорректный ответ; выполнен локальный технический pre-screening.',
      'The ML service returned an invalid response; local technical pre-screening was used.',
    ))
  }
  const code = await errorCode(response)
  if (response.status === 422 && code === 'UNSUPPORTED_PROTOCOL') {
    return fallbackWithWarning(file, t(
      'Total-body модель не применяется к этому протоколу; выполнен технический pre-screening.',
      'The total-body model does not support this protocol; technical pre-screening was used.',
    ))
  }
  if (response.status === 422 && code === 'SECONDARY_CAPTURE_EXCLUDED') {
    return fallbackWithWarning(file, t(
      'Печатный RGB DICOM принят для аудита, но исключён из ML-выборки. Требуется исходная P/R-пара.',
      'The presentation RGB DICOM was accepted for audit but excluded from ML. The source P/R pair is required.',
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
