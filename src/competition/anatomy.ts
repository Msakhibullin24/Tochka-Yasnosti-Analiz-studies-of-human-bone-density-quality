export type AnatomicalAssessment = {
  complete: boolean
  projection: { value: string; status: string; reason: string; declared_view_position: string }
  anatomy: { landmarks: { name: string; status: string; points: number[][]; verified: boolean }[];
    stability?: { status: string; unstable_landmarks: string[]; measurements: Record<string, { delta: number | null; limit: number; unstable: boolean }> } }
  source_roi: { status: string; issues: string[]; rois: { id: string; source: string; contours: { points: number[][]; depth: number }[] }[];
    checks: { roi_id: string; margin_status: string; anatomical_validity: string; margins_mm: Record<string, number> | null; candidate_landmarks_inside: Record<string, boolean | null>; reason: string }[] }
}
