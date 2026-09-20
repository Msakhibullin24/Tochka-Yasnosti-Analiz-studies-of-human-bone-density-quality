import type { Geometry } from './geometry'

export type Followup = { id: string; kind: 'markup' | 'reprocess' | 'second_opinion' | 'other'; question: string; state: 'open' | 'done'; resolution: string }
export type Draft = { baseRevision: number; author: string; comment: string; quality: '0' | '1' | 'unknown'; violations: string[]; geometry: Geometry[]; followups: Followup[] }
const prefix = 'osseo.draft.v1:'
// Each page owns a slot; a second tab cannot overwrite its unsaved work.
const writer = crypto.randomUUID()
export function writeDraft(base: string, data: Draft): string | null {
  const key = `${prefix}${base}:${writer}`
  try { localStorage.setItem(key, JSON.stringify({ data, at: Date.now() })); return key } catch { return null }
}
export function removeDraft(key: string | null): void {
  try { if (key) localStorage.removeItem(key) } catch { /* browser storage unavailable */ }
}
export function readDraft(base: string): { key: string; data: Draft } | null {
  try {
    const candidates = Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index)).filter((key): key is string => key !== null).filter(key => key.startsWith(`${prefix}${base}:`)).flatMap(key => {
      try {
        const { data, at } = JSON.parse(localStorage.getItem(key) || '')
        if (!Number.isFinite(at) || !Number.isInteger(data.baseRevision) || data.baseRevision < 0 || typeof data.author !== 'string' || typeof data.comment !== 'string' || !['0', '1', 'unknown'].includes(data.quality) ||
          !Array.isArray(data.violations) || !data.violations.every((v: unknown) => typeof v === 'string') || !Array.isArray(data.geometry) || !Array.isArray(data.followups)) return []
        if (!data.geometry.every((g: Geometry) => g && typeof g.name === 'string' && ['point', 'line', 'polyline', 'polygon'].includes(g.kind) && (!g.note || typeof g.note === 'string') && Array.isArray(g.points) && g.points.length && g.points.every(p => Number.isFinite(p.x) && Number.isFinite(p.y) && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1))) return []
        if (!data.followups.every((a: Followup) => a && typeof a.id === 'string' && typeof a.question === 'string' && typeof a.resolution === 'string' && ['open', 'done'].includes(a.state) && ['markup', 'reprocess', 'second_opinion', 'other'].includes(a.kind))) return []
        return [{ key, data: data as Draft, at }]
      } catch { return [] }
    }).sort((a, b) => b.at - a.at)
    return candidates[0] || null
  } catch { return null }
}
