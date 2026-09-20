import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import AnatomyPanel from './AnatomyPanel'
import type { Geometry, ImageDetail } from './geometry'

it('shows absent ROI and only imports candidates after explicit action without overwriting edits', () => {
  const detail: ImageDetail = { width: 101, height: 201, region: 'hip_left', pixel_mm_x: 1, pixel_mm_y: 1, pixel_mm_source: 'PixelSpacing', geometry: {}, assessment: {
    complete: false, projection: { value: 'frontal', status: 'candidate', reason: 'Требуется проверка', declared_view_position: '' },
    anatomy: { landmarks: [{ name: 'femoral_neck', status: 'candidate', points: [[20, 40]], verified: false }, { name: 'ischium', status: 'candidate', points: [[50, 100]], verified: false }] },
    source_roi: { status: 'absent', issues: [], rois: [], checks: [] },
  } }
  const existing: Geometry = { name: 'femoral_neck', kind: 'point', points: [{ x: .8, y: .8 }] }
  const change = vi.fn()
  render(<AnatomyPanel detail={detail} source="/image.png" geometry={[existing]} onChange={change} />)
  expect(screen.getByText(/Полная анатомическая проверка не подтверждена/)).toBeInTheDocument()
  expect(screen.getByText(/Исходная ROI отсутствует/)).toBeInTheDocument()
  expect(change).not.toHaveBeenCalled()
  fireEvent.click(screen.getByText('Посмотреть кандидаты и исходную ROI'))
  fireEvent.click(screen.getByRole('button', { name: 'Добавить кандидаты в черновик' }))
  expect(change.mock.calls[0][0]).toEqual([existing, { name: 'ischium', kind: 'point', points: [{ x: .5, y: .5 }], note: 'Автоматический кандидат; анатомия не подтверждена.' }])
})
