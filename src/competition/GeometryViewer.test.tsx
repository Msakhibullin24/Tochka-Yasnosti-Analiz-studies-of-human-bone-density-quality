import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import GeometryViewer from './GeometryViewer'
import type { Geometry } from './geometry'

afterEach(() => vi.unstubAllGlobals())
it('commits one bounded drag, maps screen coordinates and cancels without a change', () => {
  vi.stubGlobal('PointerEvent', class extends MouseEvent {
    pointerId: number
    constructor(type: string, options: PointerEventInit) { super(type, options); this.pointerId = options.pointerId || 1 }
  })
  const geometry: Geometry[] = [{ name: 'roi', kind: 'point', points: [{ x: .5, y: .5 }] }]
  const onChange = vi.fn()
  render(<GeometryViewer detail={{ width: 101, height: 201, region: 'spine', pixel_mm_x: 1, pixel_mm_y: 1, pixel_mm_source: 'test', geometry: {} }}
    source="test.png" geometry={geometry} original={false} zoom={2} showLayers editable onChange={onChange} />)
  const svg = screen.getByRole('group')
  Object.defineProperty(svg, 'getScreenCTM', { value: () => ({ inverse: () => ({}) }) })
  Object.defineProperty(svg, 'createSVGPoint', { value: () => ({ x: 0, y: 0,
    matrixTransform() { return { x: (this.x - 10) / 2, y: (this.y - 20) / 2 } },
  }) })
  const handle = screen.getByRole('button')
  Object.defineProperty(handle, 'setPointerCapture', { value: vi.fn() })
  fireEvent.pointerDown(handle, { pointerId: 1, button: 0 })
  fireEvent.pointerMove(handle, { pointerId: 1, clientX: 160, clientY: 220 })
  expect(onChange).not.toHaveBeenCalled()
  expect(handle).toHaveAttribute('cx', '75')
  fireEvent.pointerUp(handle, { pointerId: 1, clientX: 1000, clientY: -100 })
  expect(onChange).toHaveBeenCalledExactlyOnceWith([{ ...geometry[0], points: [{ x: 1, y: 0 }] }])
  onChange.mockClear()
  fireEvent.pointerDown(handle, { pointerId: 2, button: 0 })
  fireEvent.pointerMove(handle, { pointerId: 2, clientX: 160, clientY: 220 })
  fireEvent.keyDown(handle, { key: 'Escape' })
  fireEvent.pointerUp(handle, { pointerId: 2, clientX: 160, clientY: 220 })
  expect(onChange).not.toHaveBeenCalled()
  expect(handle).toHaveAttribute('cx', '50')
})

it('uses physical X/Y proportions and preserves normalised keyboard edits', () => {
  const onChange = vi.fn()
  const geometry: Geometry[] = [{ name: 'axis', kind: 'point', points: [{ x: .5, y: .5 }] }]
  render(<GeometryViewer detail={{ width: 101, height: 201, region: 'spine', pixel_mm_x: .6, pixel_mm_y: 1.05, pixel_mm_source: 'organiser_v2', geometry: {} }}
    source="test.png" geometry={geometry} original={false} zoom={1} showLayers editable onChange={onChange} />)
  expect(screen.getByRole('group')).toHaveAttribute('viewBox', '0 0 60 210')
  fireEvent.keyDown(screen.getByRole('button'), { key: 'ArrowRight' })
  expect(onChange).toHaveBeenCalledWith([{ ...geometry[0], points: [{ x: .51, y: .5 }] }])
})
