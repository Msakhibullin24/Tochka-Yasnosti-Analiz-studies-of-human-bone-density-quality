import { useEffect, useState } from 'react'
import type { Geometry } from './geometry'

type Evaluation = { name: string; status: string; reason: string; proposal: Geometry | null; margins_mm?: Record<string, number>; thresholds_mm?: Record<string, number> }
export default function RoiPanel({ base, geometry, onChange }: { base: string; geometry: Geometry[]; onChange: (g: Geometry[]) => void }) {
  const payload = JSON.stringify(geometry)
  const [response, setResponse] = useState<{ payload: string; rows: Evaluation[]; error?: string } | null>(null)
  useEffect(() => {
    let alive = true
    const abort = new AbortController()
    const timer = window.setTimeout(async () => {
      try {
        const result = await fetch(`/api/v1${base}/geometry-evaluation`, { method: 'POST', signal: abort.signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ geometry: JSON.parse(payload) }) })
        if (!result.ok) throw new Error('Не удалось проверить ROI. Проверьте геометрию и соединение с сервисом.')
        const rows = await result.json()
        if (alive) setResponse({ payload, rows })
      } catch (e) { if (alive) setResponse({ payload, rows: [], error: (e as Error).message }) }
    }, 200)
    return () => { alive = false; abort.abort(); window.clearTimeout(timer) }
  }, [base, payload])
  const current = response?.payload === payload ? response : null
  return <section aria-label="Проверка отступов ROI"><h3>Отступы ROI</h3><p>Проверка выбранного полигона до границ кадра: сверху и снизу 30 мм, с выбранной стороны 20 мм. Выберите боковой край в свойствах ROI. Анатомическая корректность не определяется.</p>
    {!current && <p role="status">Расчёт отступов…</p>}{current?.error && <p role="alert">{current.error}</p>}
    {current?.rows.map(r => <article key={r.name}><h4>{r.name}</h4><p>{r.reason}</p>{r.margins_mm && <dl>{Object.entries(r.margins_mm).map(([key, value]) => <div key={key}><dt>{({ top: 'Сверху', bottom: 'Снизу', side: 'Сбоку' })[key]}</dt><dd>{value.toFixed(1)} мм · минимум {r.thresholds_mm?.[key]} мм</dd></div>)}</dl>}
      {r.proposal && <details><summary>Предложение геометрического переноса</summary><p>Точки исходного и предложенного полигона (X, Y от 0 до 1). Перенос может нарушить соответствие анатомии — проверьте его после принятия.</p><ol>{r.proposal.points.map((p, i) => <li key={i}>{geometry.find(g => g.name === r.name)?.points[i].x.toFixed(3)}, {geometry.find(g => g.name === r.name)?.points[i].y.toFixed(3)} → {p.x.toFixed(3)}, {p.y.toFixed(3)}</li>)}</ol><button onClick={() => onChange(geometry.map(g => g.name === r.name ? r.proposal! : g))}>Принять перенос {r.name}</button></details>}
    </article>)}
  </section>
}
