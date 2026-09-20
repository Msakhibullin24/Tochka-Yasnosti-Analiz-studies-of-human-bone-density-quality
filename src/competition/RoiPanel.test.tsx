import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import RoiPanel from './RoiPanel'
import type { Geometry } from './geometry'

afterEach(() => vi.unstubAllGlobals())
it('shows the server proposal but applies it only after explicit acceptance', async () => {
  const initial: Geometry = { name: 'roi', kind: 'polygon', roi_edge: 'left', points: [{ x: 0, y: 0 }, { x: .5, y: 0 }, { x: .5, y: .5 }] }
  const proposal = { ...initial, points: initial.points.map(p => ({ x: p.x + .1, y: p.y + .1 })) }
  const change = vi.fn()
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => [{ name: 'roi', status: 'fail', reason: 'Перенос', proposal, margins_mm: { top: 0, bottom: 50, side: 0 }, thresholds_mm: { top: 30, bottom: 30, side: 20 } }] })))
  render(<RoiPanel base="/image" geometry={[initial]} onChange={change} />)
  fireEvent.click(await screen.findByText('Предложение геометрического переноса'))
  expect(change).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Принять перенос roi' }))
  expect(change).toHaveBeenCalledWith([proposal])
})
it('does not offer correction when the scale is estimated', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => [{ name: 'roi', status: 'estimated', reason: 'Масштаб ориентировочный', proposal: null }] })))
  render(<RoiPanel base="/image" geometry={[]} onChange={vi.fn()} />)
  await screen.findByText('Масштаб ориентировочный')
  expect(screen.queryByRole('button', { name: /Принять перенос/ })).not.toBeInTheDocument()
})
