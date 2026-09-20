import { useRef, useState, type PointerEvent } from 'react'

import { movePoint, originalGeometry, type Geometry, type ImageDetail } from './geometry'

export default function GeometryViewer({ detail, source, geometry, original, zoom, showLayers, editable, onChange, highlight = [] }: {
  detail: ImageDetail; source: string; geometry: Geometry[]; original: boolean; zoom: number;
  showLayers: boolean; editable: boolean; highlight?: string[]; onChange: (value: Geometry[]) => void;
}) {
  const svg = useRef<SVGSVGElement>(null)
  const drag = useRef<{ pointer: number; gi: number; pi: number; initial: Geometry[]; next: Geometry[] } | null>(null)
  const [preview, setPreview] = useState<Geometry[] | null>(null)
  const displayed = original ? originalGeometry(detail) : preview || geometry
  const canEdit = editable && !original && showLayers
  function move(event: PointerEvent<SVGCircleElement>) {
    const current = drag.current
    if (!current || current.pointer !== event.pointerId) return
    const matrix = svg.current?.getScreenCTM()
    if (!matrix || !svg.current) return
    const point = svg.current.createSVGPoint()
    point.x = event.clientX; point.y = event.clientY
    const local = point.matrixTransform(matrix.inverse())
    current.next = movePoint(current.initial, current.gi, current.pi, { x: local.x / ((detail.width - 1) * detail.pixel_mm_x), y: local.y / ((detail.height - 1) * detail.pixel_mm_y) })
    setPreview(current.next)
  }
  function finish(cancel: boolean) {
    const current = drag.current
    drag.current = null; setPreview(null)
    if (!cancel && current && JSON.stringify(current.initial) !== JSON.stringify(current.next)) onChange(current.next)
  }
  return <svg ref={svg} viewBox={`0 0 ${(detail.width - 1) * detail.pixel_mm_x} ${(detail.height - 1) * detail.pixel_mm_y}`} style={{ width: `${zoom * 100}%` }} role="group" aria-label={original ? 'Машинная разметка' : 'Редактируемая разметка'}>
    <g transform={`scale(${detail.pixel_mm_x} ${detail.pixel_mm_y})`}><image href={source} x="-0.5" y="-0.5" width={detail.width} height={detail.height} role="img" aria-label="Исходное изображение с ориентирами" />
    {showLayers && displayed.map((g, gi) => <g key={g.name} opacity={highlight.length && !highlight.includes(g.name) ? .25 : 1} fill="none" stroke={original ? '#ffdc73' : '#63ffdb'} strokeWidth="1">
      {g.kind !== 'point' && <polyline points={(g.kind === 'polygon' ? [...g.points, g.points[0]] : g.points).map(p => `${p.x * (detail.width - 1)},${p.y * (detail.height - 1)}`).join(' ')} />}
      {g.points.map((p, pi) => <circle key={pi} cx={p.x * (detail.width - 1)} cy={p.y * (detail.height - 1)} r={canEdit ? 4 : g.kind === 'polyline' ? 0 : 2}
        className={canEdit ? 'qc-handle' : undefined} tabIndex={canEdit ? 0 : undefined} role={canEdit ? 'button' : undefined}
        aria-label={canEdit ? `${g.name}, точка ${pi + 1}. Перемещение стрелками` : undefined}
        onPointerDown={event => {
          if (!canEdit || event.button !== 0 || drag.current) return
          event.preventDefault(); event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId)
          drag.current = { pointer: event.pointerId, gi, pi, initial: geometry, next: geometry }
        }}
        onPointerMove={move} onPointerUp={event => { if (drag.current?.pointer === event.pointerId) { move(event); finish(false) } }}
        onPointerCancel={() => finish(true)} onLostPointerCapture={() => finish(true)}
        onKeyDown={event => {
          if (!canEdit) return
          if (event.key === 'Escape') { finish(true); return }
          const delta: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }
          if (!delta[event.key] || drag.current) return
          event.preventDefault()
          const [dx, dy] = delta[event.key], step = event.shiftKey ? 10 : 1
          onChange(movePoint(geometry, gi, pi, { x: p.x + dx * step / (detail.width - 1), y: p.y + dy * step / (detail.height - 1) }))
        }}><title>{g.name} · {pi + 1}{g.note ? `: ${g.note}` : ''}</title></circle>)}
      {g.note && g.points[0] && <text x={g.points[0].x * (detail.width - 1) + 5} y={g.points[0].y * (detail.height - 1) - 5} fill="#ffdc73" stroke="none" fontSize="8"><title>{g.note}</title>{gi + 1}</text>}
    </g>)}</g>
  </svg>
}
