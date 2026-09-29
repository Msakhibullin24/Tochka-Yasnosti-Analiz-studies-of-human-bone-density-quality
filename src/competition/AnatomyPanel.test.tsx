import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import AnatomyPanel from './AnatomyPanel'
import type { Geometry, ImageDetail } from './geometry'

it('displays a lateral candidate as a lateral projection', () => {
  const detail: ImageDetail = { width: 101, height: 201, region: 'spine', pixel_mm_x: 1, pixel_mm_y: 1, pixel_mm_source: 'PixelSpacing', geometry: {}, assessment: {
    complete: false, projection: { value: 'lateral', status: 'candidate', reason: 'Требуется проверка', declared_view_position: '' },
    anatomy: { landmarks: [] }, source_roi: { status: 'absent', issues: [], rois: [], checks: [] },
  } }
  render(<AnatomyPanel detail={detail} source="/image.png" geometry={[]} onChange={vi.fn()} />)
  expect(screen.getByText(/Проекция: предположительно боковая/)).toBeInTheDocument()
})

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

it('explains unstable candidates and excludes them from the draft', () => {
  const detail: ImageDetail = { width: 101, height: 201, region: 'hip_right', pixel_mm_x: 1, pixel_mm_y: 1, pixel_mm_source: 'PixelSpacing', geometry: {}, assessment: {
    complete: false, projection: { value: 'unknown', status: 'undetermined', reason: 'Требуется проверка', declared_view_position: '' },
    anatomy: { landmarks: [{ name: 'femoral_neck', status: 'unstable', points: [], verified: false }, { name: 'ischium', status: 'candidate', points: [[50, 100]], verified: false }],
      stability: { status: 'needs_review', unstable_landmarks: ['femoral_neck'], measurements: {} } },
    source_roi: { status: 'absent', issues: [], rois: [], checks: [] },
  } }
  const change = vi.fn()
  render(<AnatomyPanel detail={detail} source="/image.png" geometry={[]} onChange={change} />)
  expect(screen.getByText(/Шейка бедра: нестабильный кандидат/)).toBeInTheDocument()
  expect(screen.getByText(/не предлагаются для добавления/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Добавить кандидаты в черновик' }))
  expect(change.mock.calls[0][0].map((item: Geometry) => item.name)).toEqual(['ischium'])
})

it('does not replace unavailable neck axis with a bone angle or claim soft tissue is verified', () => {
  const detail: ImageDetail = { width: 101, height: 201, region: 'hip_right', pixel_mm_x: .6, pixel_mm_y: 1.05, pixel_mm_source: 'PixelSpacing', geometry: {}, assessment: {
    complete: false, projection: { value: 'unknown', status: 'undetermined', reason: '', declared_view_position: '' },
    anatomy: { landmarks: [] }, source_roi: { status: 'extracted', issues: [], rois: [], checks: [], neck_geometry: {
      status: 'evaluated', checks: [{ roi_id: 'neck', status: 'partial', deviation_from_perpendicular_deg: null,
        soft_tissue_sides: { both_sides_have_candidate_pixels: true },
        structure_exclusions: { greater_trochanter: { status: 'unavailable' } } }],
    } },
  } }
  render(<AnatomyPanel detail={detail} source="/image.png" geometry={[]} onChange={vi.fn()} />)
  expect(screen.getByText(/Перпендикулярность не оценена/)).toBeInTheDocument()
  expect(screen.getByText(/Это не подтверждает наличие мягкой ткани/)).toBeInTheDocument()
  expect(screen.getByText(/Большой вертел: контур структуры недоступен/)).toBeInTheDocument()
})
