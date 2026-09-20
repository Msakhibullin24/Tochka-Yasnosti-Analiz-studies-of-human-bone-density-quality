import { afterEach, expect, it, vi } from 'vitest'
import { readDraft, removeDraft, writeDraft, type Draft } from './drafts'

const data: Draft = { baseRevision: 3, author: 'Автор', comment: 'Вопрос', quality: 'unknown', violations: [], geometry: [], followups: [] }
afterEach(() => { vi.restoreAllMocks(); localStorage.clear() })
it('isolates images and preserves the base revision for conflict detection', () => {
  const key = writeDraft('/jobs/a/images/b', data)
  expect(readDraft('/jobs/a/images/b')?.data).toEqual(data)
  expect(readDraft('/jobs/a/images/c')).toBeNull()
  removeDraft(key)
  expect(readDraft('/jobs/a/images/b')).toBeNull()
})
it('ignores malformed browser data and reports failed writes', () => {
  localStorage.setItem('osseo.draft.v1:/jobs/a/images/b:old', '{')
  expect(readDraft('/jobs/a/images/b')).toBeNull()
  vi.spyOn(localStorage, 'setItem').mockImplementation(() => { throw new Error('Quota exceeded') })
  expect(writeDraft('/jobs/a/images/b', data)).toBeNull()
})
