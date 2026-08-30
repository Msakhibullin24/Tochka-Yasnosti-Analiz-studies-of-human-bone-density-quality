import type { DatasetProtocol } from './dataset'

const API_BASE = (import.meta.env.VITE_ANALYSIS_API_URL || '/api/v1').replace(/\/$/, '')

export type LongitudinalProtocol = Extract<DatasetProtocol, 'spine_pa' | 'hip_left' | 'hip_right'>
export type MeasurementSite = 'l1-l4' | 'total-hip' | 'femoral-neck'
export type ReviewStatus = 'accept' | 'review' | 'reject'

export interface LongitudinalMeasurement {
  schemaVersion: '2.0.0'
  studyId: string
  patientGroupId: string
  acquiredOn: string
  protocol: LongitudinalProtocol
  deviceGroup: string
  facilityId: string
  operatorGroup?: string
  qualityStatus: ReviewStatus
  positioningStatus: ReviewStatus
  roiStatus: ReviewStatus
  source: 'vendor-structured' | 'dicom-sr' | 'manual-verified'
  sourceReference: string
  expertConfirmed: true
  confirmedBy: string
  sites: Array<{ site: MeasurementSite; bmd: number }>
  recordedAt?: string
}

export interface LscProfile {
  schemaVersion: '2.0.0'
  profileId: string
  version: string
  facilityId: string
  deviceGroup: string
  operatorGroup?: string
  protocol: LongitudinalProtocol
  validFrom: string
  validTo: string
  sites: Array<{ site: MeasurementSite; percent: number }>
  precisionStudyReference: string
  approvedBy: string
  status: 'active' | 'retired'
  recordedAt?: string
}

export interface CrossCalibration {
  calibrationId: string
  fromDeviceGroup: string
  toDeviceGroup: string
  validFrom: string
  validTo: string
  sites: MeasurementSite[]
  status: 'active' | 'retired'
}

export interface LongitudinalComparison {
  schemaVersion: '2.0.0'
  comparisonId: string
  engineVersion: string
  decisionBasis: 'deterministic-quality-gates'
  baselineStudyId: string
  currentStudyId: string
  patientGroupId: string
  baselineDate: string
  currentDate: string
  intervalMonths: number
  lscProfileId: string
  lscProfileVersion: string
  crossCalibrationId?: string | null
  status: 'comparable' | 'review' | 'not-comparable'
  assumed: false
  checks: Array<{ id: string; label: string; status: 'pass' | 'review' | 'block'; passed: boolean; critical: boolean; detail: string }>
  sites: Array<{ site: MeasurementSite; baselineBmd: number; currentBmd: number; absoluteChange: number; percentChange: number; lscPercent: number; status: 'significant-gain' | 'stable' | 'significant-loss' | 'not-comparable' }>
  recommendationCode: string
  comparedAt: string
}

class LongitudinalApiError extends Error {
  constructor(public readonly code: string, message: string) { super(message); this.name = 'LongitudinalApiError' }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    let detail: { code?: string; message?: string } = {}
    try { detail = (await response.json() as { detail?: typeof detail }).detail ?? {} } catch { /* non-JSON response */ }
    throw new LongitudinalApiError(detail.code ?? 'LONGITUDINAL_REQUEST_FAILED', detail.message ?? `HTTP ${response.status}`)
  }
  return response.json() as Promise<T>
}

const jsonPut = <T>(path: string, value: unknown) => request<T>(path, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) })

export const getTimeline = (patientGroupId: string) => request<{ patientGroupId: string; measurements: LongitudinalMeasurement[]; count: number }>(`/longitudinal/patients/${encodeURIComponent(patientGroupId)}/timeline`)
export const getLscProfiles = () => request<{ profiles: LscProfile[]; count: number }>('/longitudinal/lsc-profiles')
export const getCrossCalibrations = () => request<{ calibrations: CrossCalibration[]; count: number }>('/longitudinal/cross-calibrations')
export const saveMeasurement = (value: LongitudinalMeasurement) => jsonPut<LongitudinalMeasurement>(`/longitudinal/measurements/${encodeURIComponent(value.studyId)}`, value)
export const saveLscProfile = (value: LscProfile) => jsonPut<LscProfile>(`/longitudinal/lsc-profiles/${encodeURIComponent(value.profileId)}`, value)
export const compareLongitudinal = (value: { baselineStudyId: string; currentStudyId: string; lscProfileId: string; crossCalibrationId?: string }) => request<LongitudinalComparison>('/longitudinal/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) })
