import type { AnatomicalAssessment } from './anatomy'
export type Point = { x: number; y: number }
export type Geometry = { name: string; kind: 'point' | 'line' | 'polyline' | 'polygon'; points: Point[]; note?: string; roi_edge?: 'left' | 'right' | null }
export type ImageDetail = { assessment?: AnatomicalAssessment; width: number; height: number; region: string; pixel_mm_x: number; pixel_mm_y: number; pixel_mm_source: string; geometry: Record<string, number[][]> }

export function originalGeometry(detail: ImageDetail): Geometry[] {
  if (detail.width <= 1 || detail.height <= 1) return []
  return Object.entries(detail.geometry)
    .filter(([, pts]) => pts.length > 0 && pts.length <= 128 && pts.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && x >= 0 && y >= 0 && x < detail.width && y < detail.height))
    .map(([name, pts]) => ({ name, kind: pts.length === 1 ? 'point' : pts.length === 2 ? 'line' : 'polyline',
      points: pts.map(([x, y]) => ({ x: x / (detail.width - 1), y: y / (detail.height - 1) })) }))
}

export function movePoint(geometry: Geometry[], gi: number, pi: number, point: Point): Geometry[] {
  return geometry.map((g, index) => index !== gi ? g : { ...g, points: g.points.map((p, index) => index !== pi ? p : {
    x: Math.max(0, Math.min(1, point.x)), y: Math.max(0, Math.min(1, point.y)),
  }) })
}
