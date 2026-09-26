export type Row = Record<string, string>
export const reviewLabels: Record<string, string> = {
  unreviewed: 'Не проверено', draft: 'Черновик', confirmed: 'Подтверждено',
  not_evaluable: 'Невозможно оценить', unavailable: 'Проверка недоступна',
}
export const filters = ['all', 'pending', 'violations', 'failures', 'completed', 'actions', 'second_opinion'] as const
export type QueueFilter = typeof filters[number]
export function rowKey(row: Row): string { return row.row_id || row.path_to_file }
export function reviewStatus(row: Row): string {
  return row.processing_status !== 'Success' ? 'unavailable' : row.review_status || 'unreviewed'
}
export function needsReview(row: Row): boolean {
  return ['unreviewed', 'draft'].includes(reviewStatus(row))
}
export function visibleRows(rows: Row[], filter: QueueFilter, search: string): Row[] {
  const query = search.trim().toLocaleLowerCase('ru')
  return rows.filter(row => {
    const matches = filter === 'all' || (filter === 'pending' && needsReview(row)) ||
      (filter === 'violations' && row.processing_status === 'Success' && row.quality_class === '1') ||
      (filter === 'failures' && row.processing_status !== 'Success') ||
      (filter === 'actions' && Number(row.open_actions) > 0) ||
      (filter === 'second_opinion' && row.second_opinion_requested === '1') ||
      (filter === 'completed' && ['confirmed', 'not_evaluable'].includes(reviewStatus(row)))
    return matches && [row.path_to_file, row.study_uid, row.anatomical_region, row.violation_type]
      .join(' ').toLocaleLowerCase('ru').includes(query)
  })
}
export function groupStudies(rows: Row[]) {
  const groups = new Map<string, { key: string; identified: boolean; rows: Row[] }>()
  rows.forEach((row, index) => {
    // A directory name is not evidence of study identity.
    const key = row.study_uid ? `uid:${row.study_uid}` : `unknown:${index}`
    if (!groups.has(key)) groups.set(key, { key, identified: Boolean(row.study_uid), rows: [] })
    groups.get(key)!.rows.push(row)
  })
  return [...groups.values()]
}
export function nextPending(rows: Row[], selected: Row | null): Row | undefined {
  const index = selected ? rows.findIndex(r => rowKey(r) === rowKey(selected)) : -1
  return [...rows.slice(index + 1), ...rows.slice(0, Math.max(index, 0))].find(needsReview)
}

type Position = { jobId: string; rowId: string; filter: QueueFilter; zoom: number }
const storageKey = 'osseo.worklist.v1'
export function readPosition(): Position | null {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey) || 'null')
    return value && typeof value.jobId === 'string' && typeof value.rowId === 'string' &&
      filters.includes(value.filter) && typeof value.zoom === 'number' && value.zoom >= 1 && value.zoom <= 4 ? value : null
  } catch { return null }
}
export function savePosition(value: Position): void {
  // Browser storage can be unavailable; it must never prevent reviewing a study.
  try { localStorage.setItem(storageKey, JSON.stringify(value)) } catch { /* optional preference */ }
}

export function modelVerdict(row: Row): string {
  if (row.decision_reason === 'binary_only_review') return 'Сигнал модели без установленного типа: требуется проверка'
  return row.violation_type || (row.quality_class === '1' ? 'Модель выявила нарушение; тип не установлен' : 'Модель не выявила нарушений')
}
