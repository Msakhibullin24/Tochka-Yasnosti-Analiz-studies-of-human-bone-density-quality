export type Locale = 'ru' | 'en'
export type StudyStatus = 'passed' | 'review' | 'rejected'
export type CriterionStatus = 'pass' | 'warning' | 'fail'
export type StudyType = 'spine' | 'hip'

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
  criteria: Criterion[]
  recommendation: LocalizedText
}
