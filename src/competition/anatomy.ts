import type { SpecialistAssessment } from './SpecialistPanel'

export type AnatomicalAssessment = {
  specialist_qc?: SpecialistAssessment
  specialist_outputs?: Record<string, SpecialistAssessment>
  complete: boolean
  projection: { value: string; status: string; reason: string; declared_view_position: string }
  anatomy: { landmarks: { name: string; status: string; points: number[][]; verified: boolean }[];
    stability?: { status: string; unstable_landmarks: string[]; measurements: Record<string, { delta: number | null; limit: number; unstable: boolean }> } }
  source_roi: { neck_geometry?: { status: string; checks: { roi_id: string; status: string;
    roi_to_neck_axis_angle_deg?: number | null; deviation_from_perpendicular_deg?: number | null;
    soft_tissue_sides?: { both_sides_have_candidate_pixels: boolean } | null;
    structure_exclusions?: Record<string, { status: string; candidate_intersection_pixels?: number }> }[] }; status: string; issues: string[]; rois: { id: string; source: string; contours: { points: number[][]; depth: number }[] }[];
    checks: { roi_id: string; purpose?: string; margin_status: string; anatomical_validity: string; margins_mm: Record<string, number> | null; candidate_landmarks_inside: Record<string, boolean | null>; reason: string }[] }
}
