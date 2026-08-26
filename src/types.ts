export type Locale = 'ru' | 'en'
export type StudyStatus = 'passed' | 'review' | 'rejected'
export type CriterionStatus = 'pass' | 'warning' | 'fail'
export type StudyType = 'spine' | 'hip' | 'total-body'
export type AnalysisMode = 'demo-case' | 'technical-screening' | 'research-model' | 'validated-model'
export type ComparisonStatus = 'comparable' | 'review' | 'not-comparable'
export type TrendStatus = 'significant-gain' | 'stable' | 'significant-loss' | 'not-comparable'
export type MeasurementSite = 'l1-l4' | 'total-hip' | 'femoral-neck'

export interface LocalizedText {
  ru: string
  en: string
}

export interface Criterion {
  id: string
  title: LocalizedText
  detail: LocalizedText
  status: CriterionStatus
  confidence: number
  code?: string
}

export interface Landmark {
  name: string
  x: number
  y: number
  confidence: number
  visible: boolean
}

export interface RoutingTrace {
  protocol: StudyType | 'unsupported'
  confidence: number
  source: 'dicom-rules' | 'manual-override' | 'ambiguous'
  evidence: string[]
  modelKey: string
  modelStatus: 'ready' | 'planned' | 'unsupported'
}

export interface DicomTechnicalData {
  modality: string
  rows?: number | null
  columns?: number | null
  pixelSpacing?: string | null
  photometricInterpretation?: string | null
  transferSyntaxUid?: string | null
  bitsAllocated?: number | null
}

export interface AnalysisProvenance {
  mode: AnalysisMode
  modelVersion: string
  criteriaVersion: string
  processedAt: string
  warnings: LocalizedText[]
}

export interface PrivacyStatus {
  deidentificationVerified: boolean
  burnedInAnnotation?: 'YES' | 'NO'
}

export interface TrendPoint {
  studyId: string
  date: string
  bmd: number
}

export interface SiteChange {
  site: MeasurementSite
  label: LocalizedText
  baselineBmd: number
  currentBmd: number
  absoluteChange: number
  percentChange: number
  lscPercent: number
  status: TrendStatus
  history: TrendPoint[]
}

export interface ComparabilityCheck {
  id: 'protocol' | 'device' | 'cross-calibration' | 'positioning' | 'roi'
  label: LocalizedText
  passed: boolean
  detail: LocalizedText
  critical: boolean
}

export interface LongitudinalAnalysis {
  baselineStudyId: string
  baselineDate: string
  currentDate: string
  intervalMonths: number
  status: ComparisonStatus
  confidence: number
  checks: ComparabilityCheck[]
  sites: SiteChange[]
  summary: LocalizedText
  recommendation: LocalizedText
  assumed: boolean
}

export interface Study {
  id: string
  patientId: string
  filename: string
  acquiredAt: string
  type: StudyType
  status: StudyStatus
  score: number
  confidence: number
  device: string
  institution: LocalizedText
  operator: string
  accessionNumber: string
  seriesUid: string
  studyInstanceUid?: string
  technical: DicomTechnicalData
  provenance: AnalysisProvenance
  privacy: PrivacyStatus
  previewUrl?: string
  routing?: RoutingTrace
  landmarks?: Landmark[]
  longitudinal?: LongitudinalAnalysis
  criteria: Criterion[]
  recommendation: LocalizedText
}
