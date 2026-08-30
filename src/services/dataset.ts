const API_BASE = (import.meta.env.VITE_ANALYSIS_API_URL || '/api/v1').replace(/\/$/, '')

export const resolveDatasetUrl = (url?: string): string | undefined => {
  if (!url) return undefined
  return url.startsWith('/api/v1/') ? `${API_BASE}${url.slice('/api/v1'.length)}` : url
}

export const datasetExportUrl = (format: 'coco' | 'annotations') =>
  `${API_BASE}/datasets/current/exports/${format}`

export type DatasetProtocol = 'spine_pa' | 'hip_left' | 'hip_right' | 'forearm_left' | 'forearm_right' | 'total_body' | 'unsupported'
export type DatasetSplit = 'train' | 'validation' | 'test'
export type AnnotationAction = 'accept' | 'review' | 'repeat'
export type AnnotationSeverity = 'none' | 'minor' | 'major' | 'critical'

export interface DatasetSummary {
  schemaVersion: string
  loaderVersion?: string
  studyCount: number
  patientGroupCount?: number
  annotatedStudyCount: number
  annotationCount: number
  reviewRequiredCount: number
  protocols: Record<string, number>
  splits: Record<string, number>
  phaseSemantics: 'unverified'
  technicalQc?: { trainingCandidateCount?: number; expertReviewCount?: number }
}

export interface ReadinessGate {
  id: string
  label: string
  status: 'pass' | 'warn' | 'block'
  critical: boolean
  evidence: string
  nextAction: string
}

export interface ReadinessReport {
  schemaVersion: string
  policyVersion: string
  intendedUse: { ru: string; en: string }
  datasetVersion: string
  stage: 'research' | 'silent-pilot-ready'
  criticalBlockerCount: number
  warningCount: number
  gates: ReadinessGate[]
  integrity: {
    status: 'pass' | 'fail'
    studyCount: number
    patientGroupCount: number
    processedAssetCount: number
    blockerCount: number
    warningCount: number
  }
  agreement: {
    annotationCoverage: number
    doubleReadCoverage: number
    pendingAdjudicationCount: number
    overallAction: { comparisonCount: number; percentAgreement: number | null; cohenKappa: number | null }
  }
  audit: { valid: boolean; eventCount: number; headHash: string }
}

export interface ProcessedMetric {
  tag: string
  width: number
  height: number
  p01: number
  p99: number
  clippedFraction: number
}

export interface DatasetAsset { name: string; tag: string; url: string }
export interface RawChannel { index: number; label: string; url: string }

export interface TechnicalQc {
  baselineVersion: string
  score: number
  trainingCandidate: boolean
  requiresExpertReview: boolean
  flags: string[]
  processed: Array<{ tag: string; dynamicRangeFraction: number; clippedFraction: number; aspectRatio: number }>
  raw: { saturatedFraction: number; phaseSemantics: 'unverified' }
  limitations: string[]
}

export interface AnnotationDefect {
  code: string
  present: boolean
  severity: AnnotationSeverity
  action?: AnnotationAction
  confidence?: number
  comment?: string
}

export interface AnnotationLandmark { name: string; x: number; y: number; visible: boolean }
export interface AnnotationRegion { name: string; geometryType: 'box' | 'polygon' | 'polyline'; points: [number, number][] }

export interface AnnotationDocument {
  schemaVersion: '1.0.0'
  studyId: string
  patientGroupId: string
  deviceGroup?: string
  protocol: 'spine_pa' | 'hip_left' | 'hip_right' | 'total_body' | 'unsupported'
  evaluable: boolean
  notEvaluableReason: string
  overallAction: AnnotationAction
  defects: AnnotationDefect[]
  landmarks: AnnotationLandmark[]
  regions: AnnotationRegion[]
  expert: {
    readerId: string
    readIndex: number
    confidence: 'low' | 'medium' | 'high'
    createdAt: string
    adjudicated: boolean
    comment: string
  }
}

export interface DatasetStudy {
  schemaVersion: string
  loaderVersion: string
  studyId: string
  patientGroupId: string
  deviceGroup: string
  protocol: DatasetProtocol
  protocolCode: string
  acquisitionYear?: string | null
  softwareVersion?: string | null
  split?: DatasetSplit
  processedImages: ProcessedMetric[]
  raw: {
    shape: [number, number, 6]
    transmissionCount: 6
    phaseSemantics: 'unverified'
    saturatedFraction: number
    phases: Array<{ index: number; minimum: number; maximum: number; p01: number; p99: number }>
  }
  quality: { reviewRequired: boolean; flags: string[] }
  technicalQc?: TechnicalQc
  assets: DatasetAsset[]
  rawChannels: RawChannel[]
  annotationCount: number
  annotations?: AnnotationDocument[]
}

export class DatasetApiError extends Error {
  constructor(public readonly code: string, message: string, public readonly status: number) {
    super(message)
    this.name = 'DatasetApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timeout = globalThis.setTimeout(() => controller.abort(), 15_000)
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal })
  } catch (reason) {
    const timedOut = reason instanceof DOMException && reason.name === 'AbortError'
    throw new DatasetApiError(
      timedOut ? 'DATASET_REQUEST_TIMEOUT' : 'DATASET_NETWORK_ERROR',
      timedOut ? 'Dataset service did not respond within 15 seconds' : 'Dataset service is unreachable',
      0,
    )
  } finally {
    globalThis.clearTimeout(timeout)
  }
  if (!response.ok) {
    let detail: { code?: string; message?: string } = {}
    try { detail = (await response.json() as { detail?: typeof detail }).detail ?? {} } catch { /* no JSON body */ }
    throw new DatasetApiError(detail.code ?? 'DATASET_REQUEST_FAILED', detail.message ?? `HTTP ${response.status}`, response.status)
  }
  return response.json() as Promise<T>
}

export const getDatasetSummary = () => request<DatasetSummary>('/datasets/current')

export const getDatasetStudies = () => request<{ studies: DatasetStudy[]; count: number }>('/datasets/current/studies')

export const getDatasetStudy = (studyId: string) => request<DatasetStudy>(`/datasets/current/studies/${encodeURIComponent(studyId)}`)

export const getExclusionRegistry = () => request<{ exclusions: Array<{ objectId: string; reason: string; byteSize: number; recordedAt: string; trainingEligible: false }>; count: number }>('/exclusions')

export const getReadinessReport = () => request<ReadinessReport>('/readiness')

export const saveDatasetAnnotation = (annotation: AnnotationDocument) => request<AnnotationDocument>(
  `/datasets/current/studies/${encodeURIComponent(annotation.studyId)}/annotations`,
  { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(annotation) },
)
