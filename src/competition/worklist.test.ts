import { afterEach, describe, expect, it } from 'vitest'
import { groupStudies, nextPending, readPosition, savePosition, visibleRows, type Row } from './worklist'

const rows: Row[] = [
  { row_id: 'a', study_uid: '1', path_to_file: 'same/a', processing_status: 'Success', review_status: 'confirmed', quality_class: '1' },
  { row_id: 'b', study_uid: '1', path_to_file: 'other/b', processing_status: 'Success', review_status: 'draft', quality_class: '0' },
  { row_id: 'c', study_uid: '', path_to_file: 'same/c', processing_status: 'Failure', quality_class: '1' },
  { row_id: 'd', study_uid: '', path_to_file: 'same/d', processing_status: 'Success', review_status: 'not_evaluable' },
]
afterEach(() => localStorage.clear())
describe('study worklist', () => {
  it('groups by study UID across directories and never guesses missing identities', () => {
    expect(groupStudies(rows).map(s => s.rows.map(r => r.row_id))).toEqual([['a', 'b'], ['c'], ['d']])
  })
  it('keeps processing failures out of violation and review queues', () => {
    expect(visibleRows(rows, 'violations', '').map(r => r.row_id)).toEqual(['a'])
    expect(visibleRows(rows, 'pending', '').map(r => r.row_id)).toEqual(['b'])
    expect(visibleRows(rows, 'completed', '').map(r => r.row_id)).toEqual(['a', 'd'])
    expect(nextPending(rows, rows[3])?.row_id).toBe('b')
    expect(nextPending(rows, rows[1])).toBeUndefined()
  })
  it('restores validated preferences and tolerates corrupt or obsolete storage', () => {
    const position = { jobId: 'job', rowId: 'b', zoom: 2, filter: 'pending' as const }
    savePosition(position)
    expect(readPosition()).toEqual(position)
    localStorage.setItem('osseo.worklist.v1', '{')
    expect(readPosition()).toBeNull()
    localStorage.setItem('osseo.worklist.v1', JSON.stringify({ ...position, zoom: 100 }))
    expect(readPosition()).toBeNull()
  })
})

it('does not display an unknown violation type as no violations', async () => {
  const { modelVerdict } = await import('./worklist')
  expect(modelVerdict({ quality_class: '1', violation_type: '' })).toContain('тип не установлен')
  expect(modelVerdict({ quality_class: '0', violation_type: '' })).toContain('не выявила')
})
