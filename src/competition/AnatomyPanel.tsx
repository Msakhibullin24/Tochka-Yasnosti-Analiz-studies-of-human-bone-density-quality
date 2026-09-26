import type { Geometry, ImageDetail } from './geometry'

const names: Record<string, string> = { Th12: 'Позвонок Th12', iliac_crest_left: 'Левый край подвздошной кости', iliac_crest_right: 'Правый край подвздошной кости', greater_trochanter: 'Большой вертел', lesser_trochanter: 'Малый вертел', ischium: 'Седалищная кость', femoral_neck: 'Шейка бедра', vertebral_body_candidates: 'Тела позвонков без нумерации' }
const roiStatus: Record<string, string> = { absent: 'Исходная ROI отсутствует в поддерживаемых DICOM-структурах', unavailable: 'Исходную ROI не удалось прочитать', partial: 'Исходная ROI прочитана частично', extracted: 'Исходная ROI извлечена' }
const marginStatus: Record<string, string> = { pass: 'Геометрические отступы соблюдены', fail: 'Геометрические отступы не соблюдены', unavailable: 'Отступы невозможно подтвердить', not_applicable: 'Правило охвата 30/30/20 мм к этой ROI не применяется' }
const roiPurpose: Record<string, string> = { femoral_neck: 'Шейка бедра', total_hip: 'Общая область бедра', scan_coverage: 'Охват сканирования', unknown: 'Назначение не указано' }

export default function AnatomyPanel({ detail, source, geometry, onChange }: { detail: ImageDetail; source: string; geometry: Geometry[]; onChange: (g: Geometry[]) => void }) {
  const a = detail.assessment
  if (!a) return <section aria-label="Анатомическая проверка"><h3>Анатомическая проверка</h3><p>Этот результат создан до добавления проверки. Загрузите исходный пакет повторно в новое задание.</p></section>
  const candidates = a.anatomy.landmarks.filter(l => l.points.length && l.points.length <= 128 && l.points.every(([x,y]) => x >= 0 && x < detail.width && y >= 0 && y < detail.height))
  const addable = candidates.filter(l => !geometry.some(g => g.name === l.name))
  function addCandidates() {
    onChange([...geometry, ...addable.map(l => ({ name: l.name, kind: l.points.length === 1 ? 'point' as const : 'polyline' as const,
      points: l.points.map(([x,y]) => ({ x: x / (detail.width - 1), y: y / (detail.height - 1) })), note: 'Автоматический кандидат; анатомия не подтверждена.' }))])
  }
  return <section aria-label="Анатомическая проверка">
    <h3>Анатомическая проверка</h3>
    <p><strong>Полная анатомическая проверка не подтверждена.</strong> Класс качества модели не означает, что все ориентиры и разметка проверены.</p>
    <p>Проекция: {a.projection.status === 'candidate' && a.projection.value === 'frontal' ? 'предположительно фронтальная' : 'не определена по изображению'}. {a.projection.reason}</p>
    {a.projection.declared_view_position && <p>Проекция в DICOM: {a.projection.declared_view_position}.</p>}
    {a.anatomy.stability?.status === 'needs_review' && <p>Часть ориентиров или измерений бедра меняется при технической проверке яркости. Эти точки не предлагаются для добавления; проверьте снимок вручную.</p>}
    {a.anatomy.stability?.status === 'unavailable' && <p>Стабильность ориентиров в миллиметрах не проверена: масштаб снимка не подтверждён.</p>}
    <ul>{a.anatomy.landmarks.map(l => <li key={l.name}>{names[l.name] || l.name}: {l.status === 'candidate' ? 'найден кандидат, требуется проверка' : l.status === 'unstable' ? 'нестабильный кандидат — скрыт' : 'не локализован'}</li>)}</ul>
    <details><summary>Посмотреть кандидаты и исходную ROI</summary>
      <p>Жёлтые точки — кандидаты ориентиров. Голубые контуры — разметка из DICOM. Ориентация исходного снимка сохранена.</p>
      <svg viewBox={`0 0 ${detail.width} ${detail.height}`} style={{ width: '100%', maxHeight: 500, background: '#111' }} role="img" aria-label="Исходное изображение с кандидатами ориентиров и контурами ROI">
        <image href={source} width={detail.width} height={detail.height} />
        {a.source_roi.rois.map(roi => <path key={roi.id} d={roi.contours.map(c => c.points.map(([x,y],i) => `${i ? 'L' : 'M'}${x} ${y}`).join(' ') + ' Z').join(' ')} fill="none" stroke="#67e8f9" strokeWidth={1.5}><title>{roi.source}</title></path>)}
        {candidates.flatMap(l => l.points.map(([x,y],i) => <circle key={`${l.name}-${i}`} cx={x} cy={y} r={3} fill="#fde047" stroke="#111"><title>{names[l.name] || l.name}: кандидат</title></circle>))}
      </svg>
      <button type="button" disabled={!addable.length} onClick={addCandidates}>Добавить кандидаты в черновик</button>
      <p>Существующие ориентиры не заменяются. Проверьте и скорректируйте добавленные точки перед сохранением.</p>
    </details>
    <p><strong>{roiStatus[a.source_roi.status] || 'Состояние ROI не определено'}</strong></p>
    {!!a.source_roi.issues.length && <details><summary>Причины ограничений чтения ROI</summary><ul>{a.source_roi.issues.map((issue,i) => <li key={i}>{issue}</li>)}</ul></details>}
    {a.source_roi.checks.map(check => <article key={check.roi_id}><h4>Исходная ROI</h4><p>Назначение: {roiPurpose[check.purpose || 'unknown'] || check.purpose}.</p><p>{marginStatus[check.margin_status]}</p>
      {check.margins_mm && <p>Отступы: сверху {check.margins_mm.top} мм, снизу {check.margins_mm.bottom} мм, сбоку {check.margins_mm.side} мм.</p>}
      <ul>{Object.entries(check.candidate_landmarks_inside).map(([name,inside]) => <li key={name}>{names[name] || name}: {inside === null ? 'не локализован' : inside ? 'кандидат внутри ROI' : 'кандидат вне ROI'}</li>)}</ul><p>{check.reason}</p></article>)}
  </section>
}
