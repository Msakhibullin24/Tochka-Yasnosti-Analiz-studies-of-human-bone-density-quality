import { useEffect, useRef, useState } from 'react'
import './competition.css'
import RoiPanel from './competition/RoiPanel'
import AnatomyPanel from './competition/AnatomyPanel'
import SpecialistPanel from './competition/SpecialistPanel'
import ModelRegistry from './competition/ModelRegistry'
import CriteriaPanel from './competition/CriteriaPanel'
import { readDraft, writeDraft, removeDraft, type Followup } from './competition/drafts'
import GeometryViewer from './competition/GeometryViewer'
import { originalGeometry, type Geometry, type ImageDetail as Detail } from './competition/geometry'
import { modelVerdict, groupStudies, needsReview, nextPending, readPosition, reviewLabels, reviewStatus, rowKey, savePosition, visibleRows, type QueueFilter, type Row } from './competition/worklist'

type Job = { originals_available?: boolean; id: string; status: string; done: number; total: number; created_at: number; error?: string; summary?: { submission_available?: boolean; submission_valid?: boolean; files: number; success: number; failure: number } }
type Review = { revision: number; author: string; status: 'draft' | 'confirmed' | 'not_evaluable'; quality_class: 0 | 1 | null; violations: string[]; comment: string; geometry: Geometry[]; followups?: Followup[]; measurements?: Record<string, { length_mm: number; angle_from_vertical_deg: number }> }
type Catalog = { criteria: Record<string, string>; regions: Record<string, string[]> }
const API = '/api/v1'
const statuses: Record<string, string> = { queued: 'В очереди', running: 'Обработка', finished: 'Обработано', failed: 'Ошибка', cancelled: 'Отменено' }
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(API + path, options)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    const detail = body.detail
    throw Object.assign(new Error(response.status === 409 ? 'Версия на сервере изменилась. Сравните её с вашей рабочей копией.' :
      typeof detail === 'string' ? detail : 'Не удалось выполнить действие. Проверьте поля и повторите попытку.'), { status: response.status })
  }
  return response.json()
}
function parse<T>(value: string | undefined, fallback: T): T { try { return JSON.parse(value || '') as T } catch { return fallback } }

export default function CompetitionWorkspace() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [job, setJob] = useState<Job | null>(null)
  const [rows, setRows] = useState<Row[]>([])
  const [selected, setSelected] = useState<Row | null>(null)
  const [files, setFiles] = useState<File[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [search, setSearch] = useState('')
  const [position] = useState(readPosition)
  const [filter, setFilter] = useState<QueueFilter>(position?.filter || 'all')
  const [zoom, setZoom] = useState(position?.zoom || 1)
  const [catalog, setCatalog] = useState<Catalog>({ criteria: {}, regions: {} })
  const [hasEdits, setHasEdits] = useState(false)
  const [ready, setReady] = useState('Проверка сервиса…')
  const selection = useRef(0)
  const refresh = () => request<Job[]>('/jobs?limit=200').then(setJobs)
  useEffect(() => {
    let alive = true
    Promise.all([request<Job[]>('/jobs?limit=200'), request<Catalog>('/catalog'), request<{ ready: boolean }>('/ready')])
      .then(async ([history, labels, health]) => {
        if (!alive) return
        setJobs(history); setCatalog(labels); setReady(health.ready ? 'Сервис готов · локальная обработка' : 'Сервис не готов')
        const saved = readPosition()
        if (selection.current !== 0) return
        const current = history.find(item => item.id === saved?.jobId) || history[0]
        if (!current) return
        const token = ++selection.current
        setJob(current)
        if (['running', 'queued'].includes(current.status)) return
        const result = await request<Row[]>(`/jobs/${current.id}/worklist`)
        if (alive && token === selection.current) {
          setRows(result)
          setSelected(result.find(r => current.id === saved?.jobId && rowKey(r) === saved?.rowId) || result.find(needsReview) || result[0] || null)
        }
      })
      .catch(e => { if (alive) { setError(e.message); setReady('Нет подключения к сервису') } })
    return () => { alive = false }
  }, [])
  const jobId = job?.id
  const jobStatus = job?.status
  useEffect(() => {
    if (!jobId || !['running', 'queued'].includes(jobStatus || '')) return
    let alive = true
    const timer = window.setInterval(() => {
      request<Job>(`/jobs/${jobId}`).then(async current => {
        if (!alive) return
        if (!['running', 'queued'].includes(current.status)) {
          const result = await request<Row[]>(`/jobs/${jobId}/worklist`)
          if (alive) { setRows(result); setSelected(result.find(needsReview) || result[0] || null); setJob(current) }
          refresh().catch(e => { if (alive) setError(e.message) })
        } else setJob(current)
      }).catch(e => { if (alive) setError(e.message) })
    }, 700)
    return () => { alive = false; window.clearInterval(timer) }
  }, [jobId, jobStatus])
  function leaveReview() {
    if (hasEdits && !window.confirm('Есть несохранённые изменения. Перейти без сохранения?')) return false
    setHasEdits(false); return true
  }
  async function reprocessJob() {
    if (!job || !leaveReview()) return
    setBusy(true)
    try {
      const next = await request<Job>(`/jobs/${job.id}/reprocess`, { method: 'POST' })
      setJob(next); setRows([]); setSelected(null); await refresh()
    } catch (e) { setError(e instanceof Error ? e.message : 'Не удалось запустить повторную обработку') }
    finally { setBusy(false) }
  }
  async function openJob(current: Job) {
    if (!leaveReview()) return
    const token = ++selection.current
    setJob(current); setRows([]); setSelected(null); setError('')
    if (['running', 'queued'].includes(current.status)) return
    try {
      const result = await request<Row[]>(`/jobs/${current.id}/worklist`)
      if (token !== selection.current) return
      setRows(result); setSelected(result.find(needsReview) || result[0] || null)
    } catch (e) { if (token === selection.current) setError((e as Error).message) }
  }
  async function upload() {
    if (!leaveReview()) return
    setBusy(true); setError('')
    try {
      const body = new FormData(); files.forEach(f => body.append('files', f))
      const current = await request<Job>('/batch', { method: 'POST', body })
      await openJob(current); await refresh(); setFiles([])
    } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  useEffect(() => {
    if (job) savePosition({ jobId: job.id, rowId: selected ? rowKey(selected) : '', filter, zoom })
  }, [job, selected, filter, zoom])
  const visible = visibleRows(rows, filter, search)
  const studies = groupStudies(rows)
  const visibleKeys = new Set(visible.map(rowKey))
  const pending = rows.filter(needsReview).length
  const next = nextPending(visible, selected)
  function selectRow(row: Row) { if (leaveReview()) setSelected(row) }
  function savedReview(review: Review) {
    if (!selected) return
    const updated = { ...selected, review_status: review.status, review_revision: String(review.revision), open_actions: String((review.followups || []).filter(a => a.state === 'open').length), second_opinion_requested: (review.followups || []).some(a => a.kind === 'second_opinion' && a.state === 'open') ? '1' : '0' }
    setRows(current => current.map(r => rowKey(r) === rowKey(updated) ? updated : r))
    setSelected(updated)
    setHasEdits(false)
  }
  const finished = Boolean(job && ['finished', 'cancelled'].includes(job.status))
  const uploadPanel = <section className="qc-upload" aria-labelledby="upload-title">
    <div><h1 id="upload-title">Проверка исследований</h1><p>DICOM-файлы или один ZIP-архив</p></div>
    <label className="qc-file-input">Выбрать файлы<input type="file" multiple onChange={e => setFiles(Array.from(e.target.files || []))} /></label>
    <span className="qc-file-count" aria-live="polite">{files.length ? `Выбрано: ${files.length}` : 'Файлы не выбраны'}</span>
    <button className="primary-button" disabled={busy || !files.length} onClick={upload}>{busy ? 'Загрузка…' : 'Начать проверку'}</button>
  </section>
  return <div className="qc-workspace">
    <a href="#qc-main" className="skip-link">К исследованиям</a>
    <header className="qc-header"><div><strong>Osseo</strong><span>Контроль качества DXA</span></div><span className="qc-service-status" role="status">{ready}</span></header>
    <main id="qc-main">
      {error && <p className="qc-error" role="alert">{error}</p>}
      {job ? <details className="qc-new-batch"><summary>Загрузить новый пакет</summary>{uploadPanel}</details> : uploadPanel}
      <details className="qc-batches" open={!job}>
        <summary>Пакеты <span>{jobs.length}</span>{job && <small>{new Date(job.created_at * 1000).toLocaleString('ru-RU')} · {statuses[job.status] || job.status}</small>}</summary>
        <div className="qc-history"><button className="qc-text-button" onClick={() => refresh().catch(e => setError(e.message))}>Обновить</button>
          {!jobs.length && <p>Загрузите первый пакет, чтобы начать проверку.</p>}
          {jobs.map(j => <button key={j.id} className={job?.id === j.id ? 'qc-selected' : ''} aria-pressed={job?.id === j.id} onClick={() => openJob(j)}>
            <strong>{new Date(j.created_at * 1000).toLocaleString('ru-RU')}</strong><span>{statuses[j.status] || j.status} · {j.summary?.files ?? j.total} файлов</span>
          </button>)}
          {jobs.length === 200 && <p>Показаны последние 200 пакетов.</p>}
        </div>
      </details>
      <ModelRegistry />
      {!job && <section className="qc-empty"><h2>Выберите пакет или загрузите новый</h2><p>После обработки здесь появятся очередь снимков и форма решения.</p></section>}
      {job && <>
        <section className="qc-jobbar" aria-label="Текущий пакет">
          <div><strong>{statuses[job.status] || job.status}</strong><span>{job.total > 0 ? `${job.done} из ${job.total} файлов` : 'Нет файлов'}</span></div>
          {finished && <span className="qc-job-counts" role="status">{pending} требуют проверки · {rows.filter(r => r.processing_status !== 'Success').length} ошибок</span>}
          {finished && <button className="primary-button" disabled={!next} onClick={() => next && selectRow(next)}>Следующий снимок</button>}
          <details className="qc-menu"><summary>Действия</summary><div>
            {job.originals_available && finished && <><a href={`${API}/jobs/${job.id}/source.zip`} download>Скачать исходные DICOM</a><button disabled={busy} onClick={reprocessJob}>Повторить анализ</button></>}
            {finished && <nav aria-label="Экспорт результатов">
              {(['finished', 'cancelled'].includes(job.status) ? ['results.csv', 'results.xlsx', ...(job.summary?.submission_available ? ['submission.csv', 'submission.xlsx', 'submission_validation.json'] : []), 'additional_series.zip', 'reviews.json', 'reviewed.csv', 'review-package.zip', 'report.html', 'validation.json'] : ['results.csv', 'results.xlsx', 'report.html']).map(name => <a key={name} href={`${API}/jobs/${job.id}/${name}`} download>{({ 'submission.csv': 'CSV для сдачи', 'submission.xlsx': 'XLSX для сдачи', 'submission_validation.json': 'Проверка таблицы для сдачи', 'results.csv': 'CSV модели', 'results.xlsx': 'XLSX модели', 'additional_series.zip': 'Серии DICOM', 'review-package.zip': 'Пакет экспертной проверки', 'reviews.json': 'История правок', 'reviewed.csv': 'CSV специалиста', 'report.html': 'Отчёт для печати', 'validation.json': 'Проверка формата ТЗ' })[name]}</a>)}
            </nav>}
          </div></details>
        </section>
        {job.error && <p className="qc-error">{job.error}</p>}
        {job.summary?.submission_available && !job.summary.submission_valid && <p className="qc-warning">Таблица для сдачи требует исправлений. Откройте «Действия» → «Проверка таблицы для сдачи».</p>}
        {['running', 'queued'].includes(job.status) ? <section className="qc-processing"><progress max={Math.max(1, job.total)} value={job.done} aria-label="Обработка пакета" /><p>Обработано {job.done} из {job.total}</p><button onClick={() => request(`/jobs/${job.id}/cancel`, { method: 'POST' }).catch(e => setError(e.message))}>Остановить</button></section> :
        <div className="qc-workbench">
          <aside className="qc-queue" aria-label="Очередь снимков">
            <div className="qc-queue-head"><div><h2>Снимки</h2><span>{visible.length} из {rows.length}</span></div><label>Поиск<input placeholder="Имя файла или область" value={search} onChange={e => setSearch(e.target.value)} /></label><label>Фильтр<select value={filter} onChange={e => setFilter(e.target.value as QueueFilter)}><option value="all">Все снимки</option><option value="pending">Требуют проверки</option><option value="completed">Проверка завершена</option><option value="actions">Есть открытые действия</option><option value="second_opinion">Нужно второе мнение</option><option value="violations">Нарушения ИИ</option><option value="failures">Ошибки обработки</option></select></label></div>
            <div className="qc-row-list">{studies.map((study, index) => {
              const shown = study.rows.filter(r => visibleKeys.has(rowKey(r)))
              if (!shown.length) return null
              const count = study.rows.filter(needsReview).length
              const failures = study.rows.filter(r => r.processing_status !== 'Success').length
              return <section key={study.key} className="qc-study" aria-label={`Группа ${index + 1}`}>
                <h3>{study.identified ? `Исследование ${index + 1}` : 'Файл без идентификатора исследования'}</h3>
                <p>{study.rows.length} снимков · {count ? `Ожидают проверки: ${count}` : failures ? 'Обработка неполная' : 'Проверка завершена'}{failures > 0 && ` · Ошибок: ${failures}`}</p>
                {study.identified && <details><summary>Данные исследования</summary><small>{study.rows[0].study_uid}</small><a href={`${API}/jobs/${job.id}/study-report.html?study_uid=${encodeURIComponent(study.rows[0].study_uid)}`} download>Скачать карточку</a></details>}
                {shown.map((r, i) => <button key={rowKey(r) || i} className={selected && rowKey(selected) === rowKey(r) ? 'qc-selected' : ''} onClick={() => selectRow(r)} aria-pressed={Boolean(selected && rowKey(selected) === rowKey(r))}>
                  <strong>{r.anatomical_region || 'Не обработано'}</strong><span>{r.processing_status !== 'Success' ? r.error_message : modelVerdict(r)}</span>
                  <span>{Number(r.open_actions) > 0 && `Открытых действий: ${r.open_actions} · `}{r.second_opinion_requested === '1' && 'Нужно второе мнение · '}{reviewLabels[reviewStatus(r)]}{Number(r.review_revision) > 0 && ` · Версия ${r.review_revision}`}</span><small>{r.path_to_file}</small>
                </button>)}
              </section>
            })}</div>
            {rows.length > 0 && !visible.length && <p>Нет изображений по выбранным условиям. Сбросьте фильтры, чтобы увидеть все снимки.</p>}
            {selected && !visibleKeys.has(rowKey(selected)) && <p>Открытый снимок не соответствует текущему фильтру. Он остаётся открыт для завершения работы.</p>}
          </aside>
          <div className="qc-review-pane">
            {selected?.processing_status === 'Success' && <ImageReview key={job.id + selected.row_id} jobId={job.id} row={selected} catalog={catalog} onDirty={setHasEdits} onSaved={savedReview} zoom={zoom} onZoom={setZoom} />}
            {selected?.processing_status === 'Failure' && <section className="qc-empty"><h2>Снимок не обработан</h2><p>{selected.error_message}</p><code>{selected.error_code}</code></section>}
            {!selected && <section className="qc-empty"><h2>Выберите снимок</h2><p>Результат модели и форма решения откроются здесь.</p></section>}
          </div>
        </div>}
      </>}
    </main>
  </div>
}

function ImageReview({ jobId, row, catalog, onDirty, onSaved, zoom, onZoom }: { jobId: string; row: Row; catalog: Catalog; onDirty: (dirty: boolean) => void; onSaved: (review: Review) => void; zoom: number; onZoom: (zoom: number) => void }) {
  const base = `/jobs/${jobId}/images/${row.row_id}`
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  const [detail, setDetail] = useState<Detail | null>(null)
  const [history, setHistory] = useState<Review[]>([])
  const [geometry, setGeometry] = useState<Geometry[]>([])
  const [undo, setUndo] = useState<Geometry[][]>([])
  const [redo, setRedo] = useState<Geometry[][]>([])
  const [author, setAuthor] = useState('')
  const [comment, setComment] = useState('')
  const [followups, setFollowups] = useState<Followup[]>([])
  const [baseRevision, setBaseRevision] = useState(0)
  const [recovered, setRecovered] = useState<ReturnType<typeof readDraft>>(null)
  const [draftMessage, setDraftMessage] = useState('')
  const localDraftKey = useRef<string | null>(null)
  const recoveredKey = useRef<string | null>(null)
  const [comparison, setComparison] = useState<number | null>(null)
  const [conflict, setConflict] = useState<Review | null>(null)
  const [quality, setQuality] = useState<'0' | '1' | 'unknown'>(row.quality_class as '0' | '1')
  const [violations, setViolations] = useState<string[]>(row.violation_codes?.split(';').filter(Boolean) || [])
  const [showLayers, setShowLayers] = useState(true)
  const [original, setOriginal] = useState(false)
  const [editable, setEditable] = useState(false)
  const [highlight, setHighlight] = useState<string[]>([])
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  useEffect(() => {
    let active = true
    Promise.all([request<Detail>(base), request<Review[]>(base + '/reviews')]).then(([d, h]) => {
      if (!active) return
      setDetail(d); setHistory(h); setRecovered(readDraft(base)); setBaseRevision(h.at(-1)?.revision || 0)
      const last = h.at(-1)
      if (last) { setAuthor(last.author); setComment(last.comment); setQuality(last.quality_class === null ? 'unknown' : String(last.quality_class) as '0' | '1'); setViolations(last.violations); setGeometry(last.geometry); setFollowups(last.followups || []) }
      else setGeometry(originalGeometry(d))
    }).catch(e => { if (active) setError(e.message) })
    return () => { active = false }
  }, [base])
  useEffect(() => { onDirty(dirty) }, [dirty, onDirty])
  useEffect(() => {
    const handler = (event: BeforeUnloadEvent) => { if (dirty) { event.preventDefault(); event.returnValue = '' } }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [dirty])
  useEffect(() => {
    if (!detail || !dirty || saving) return
    const key = writeDraft(base, { baseRevision, author, comment, quality, violations, geometry, followups })
    if (key) localDraftKey.current = key
    let active = true
    queueMicrotask(() => { if (active) setDraftMessage(key ? 'Рабочая копия сохранена в этом браузере. Версия решения ещё не записана.' : 'Не удалось сохранить рабочую копию в браузере. Сохраните черновик на сервере.') })
    return () => { active = false }
  }, [base, baseRevision, author, comment, quality, violations, geometry, followups, dirty, detail, saving])
  function restoreDraft() {
    if (!recovered) return
    const d = recovered.data
    setAuthor(d.author); setComment(d.comment); setQuality(d.quality); setViolations(d.violations); setGeometry(d.geometry); setFollowups(d.followups)
    setUndo([]); setRedo([]); setBaseRevision(d.baseRevision); recoveredKey.current = recovered.key; setRecovered(null); setDirty(true)
  }
  function changeGeometry(next: Geometry[]) { setUndo([...undo, geometry]); setRedo([]); setGeometry(next); setDirty(true) }
  async function save(status: Review['status']) {
    setError(''); setMessage('')
    if (!author.trim()) { setError('Укажите имя специалиста.'); return }
    if (quality === 'unknown' && !comment.trim()) { setError('Укажите причину, по которой изображение нельзя оценить.'); return }
    if (status === 'confirmed' && quality === '1' && !violations.length) { setError('Выберите тип нарушения.'); return }
    if (followups.some(a => !a.question.trim() || (a.state === 'done' && !a.resolution.trim()))) { setError('Укажите вопрос для каждого действия и результат выполненных действий.'); return }
    setSaving(true)
    try {
      const saved = await request<Review>(base + '/reviews', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ expected_revision: baseRevision, author, comment, quality_class: quality === 'unknown' ? null : Number(quality), status: status === 'draft' ? 'draft' : quality === 'unknown' ? 'not_evaluable' : status, violations: quality === '1' ? violations : [], geometry, followups }) })
      if (!mounted.current) return
      setHistory([...history, saved]); setBaseRevision(saved.revision); removeDraft(localDraftKey.current); removeDraft(recoveredKey.current); setDraftMessage(''); setRecovered(null); setDirty(false); onSaved(saved); setMessage(`Версия ${saved.revision} сохранена. ${saved.status === 'draft' ? 'Черновик.' : 'Решение специалиста зафиксировано.'}`)
    } catch (e) {
      if (!mounted.current) return
      setError((e as Error).message)
      if ((e as Error & { status?: number }).status === 409) {
        try {
          const current = await request<Review[]>(base + '/reviews')
          if (!mounted.current) return
          setHistory(current); setConflict(current.at(-1) || null); setComparison(current.at(-1)?.revision || null)
        } catch { setError('Не удалось загрузить актуальную версию. Рабочая копия сохранена в браузере; повторите сохранение после восстановления связи.') }
      }
    } finally { setSaving(false) }
  }
  const criteria = parse<Record<string, number>>(row.violation_scores, {})
  const thresholds = parse<Record<string, number>>(row.criterion_thresholds, {})
  const allowed = detail ? catalog.regions[detail.region] || [] : []
  return <section className="qc-review"><h2>{row.anatomical_region}</h2><p className="qc-path">{row.path_to_file}</p>
    {error && <p className="qc-error" role="alert">{error}</p>}
    {recovered && <aside className="qc-card" aria-label="Восстановление черновика"><p>Найдена рабочая копия в этом браузере: {recovered.data.author || 'Автор не указан'}. {recovered.data.baseRevision !== (history.at(-1)?.revision || 0) && 'На сервере уже другая версия. Восстановленную копию нельзя записать поверх неё без разрешения конфликта.'}</p><button onClick={restoreDraft}>Восстановить рабочую копию</button><button onClick={() => { removeDraft(recovered.key); setRecovered(null) }}>Удалить найденную копию</button></aside>}
    {conflict && <aside className="qc-card"><p>На сервере версия {conflict.revision}, автор {conflict.author}. Сравните её с вашей разметкой ниже. Продолжение оставит вашу рабочую копию основой новой версии; исходные версии сохранятся в истории.</p><button onClick={() => { setBaseRevision(conflict.revision); setConflict(null); setError(''); setDirty(true) }}>Продолжить с моими правками после версии {conflict.revision}</button></aside>}
    <fieldset disabled={saving || !detail} className="qc-editor-body"><div className="qc-review-grid"><div>
      <div className="qc-actions"><label><input type="checkbox" checked={showLayers} onChange={e => setShowLayers(e.target.checked)} /> Показать ориентиры</label><label>Масштаб <input type="range" min="1" max="4" step="0.25" value={zoom} onChange={e => onZoom(Number(e.target.value))} /></label></div>
      <div className="qc-actions"><button aria-pressed={original} onClick={() => { setOriginal(!original); setHighlight([]) }}>{original ? 'Показать мою разметку' : 'Показать машинную разметку'}</button>
        <label><input type="checkbox" checked={editable} disabled={original || saving} onChange={e => setEditable(e.target.checked)} /> Перемещать точки</label>
        <button onClick={() => onZoom(1)}>Сбросить масштаб</button>
      </div>
      {(original || editable) && <p role="status">{original ? 'Показана исходная машинная разметка.' : 'Точки перемещаются мышью или стрелками. Shift — шаг 10 пикселей.'}</p>}
      <div className="qc-viewer" tabIndex={0} aria-label="Просмотр изображения; при увеличении доступна прокрутка">
        {detail && <GeometryViewer key={`${original}-${editable}-${showLayers}`} detail={detail} source={API + base + '/original.png'} geometry={geometry} original={original} zoom={zoom} showLayers={showLayers} editable={editable && !saving} onChange={changeGeometry} highlight={highlight} />}

      </div>
      {detail && <details><summary>Параметры изображения</summary><p>Физические пропорции, нормализованная яркость. Пиксель: {detail.pixel_mm_x.toFixed(3)} × {detail.pixel_mm_y.toFixed(3)} мм. {detail.pixel_mm_source === 'device_default' ? 'Масштаб аппарата требует проверки.' : detail.pixel_mm_source === 'organiser_v2' ? 'Источник: разъяснения организатора V2.' : `Источник: ${detail.pixel_mm_source}.`}</p></details>}
      <details><summary>Изменить ориентиры и ROI</summary><p>Координаты относительно исходного изображения: от 0 до 1. Изменения сохраняются с экспертным решением.</p>
        <div className="qc-actions"><button disabled={!undo.length} onClick={() => { setRedo([...redo, geometry]); setGeometry(undo[undo.length - 1]); setUndo(undo.slice(0, -1)); setDirty(true) }}>Отменить изменение</button><button disabled={!redo.length} onClick={() => { setUndo([...undo, geometry]); setGeometry(redo[redo.length - 1]); setRedo(redo.slice(0, -1)); setDirty(true) }}>Повторить изменение</button>
        <button onClick={() => changeGeometry([...geometry, { name: `roi_${Date.now()}`, kind: 'polygon', points: [{ x: 0.25, y: 0.25 }, { x: 0.75, y: 0.25 }, { x: 0.75, y: 0.75 }, { x: 0.25, y: 0.75 }] }])}>Добавить ROI</button><button disabled={geometry.length >= 40} onClick={() => { changeGeometry([...geometry, { name: `note_${Date.now()}`, kind: 'point', points: [{ x: .5, y: .5 }], note: '' }]); setEditable(true); setOriginal(false) }}>Добавить отметку</button></div>
        {geometry.map((g, gi) => <fieldset key={g.name}><legend>{g.name}</legend>
          {g.kind === 'polygon' && <label>Боковой край для {g.name}<select value={g.roi_edge || ''} onChange={e => changeGeometry(geometry.map((item, index) => index === gi ? { ...item, roi_edge: (e.target.value || null) as Geometry['roi_edge'] } : item))}><option value="">Не выбран</option><option value="left">Левый край изображения</option><option value="right">Правый край изображения</option></select></label>}
          <label>Замечание к {g.name}<textarea maxLength={1000} value={g.note || ''} onChange={e => changeGeometry(geometry.map((item, index) => index === gi ? { ...item, note: e.target.value } : item))} /></label>
          {g.points.map((p, pi) => <div className="qc-coordinates" key={pi}>{(['x', 'y'] as const).map(axis => <label key={axis}>Точка {pi + 1} · {axis.toUpperCase()}<input type="number" min="0" max="1" step="0.001" value={Number(p[axis].toFixed(4))} onChange={e => { const n = Number(e.target.value); if (Number.isFinite(n) && n >= 0 && n <= 1) changeGeometry(geometry.map((item, index) => index !== gi ? item : { ...item, points: item.points.map((point, index) => index !== pi ? point : { ...point, [axis]: n }) })) }} /></label>)}</div>)}
          <button onClick={() => changeGeometry(geometry.filter((_, i) => i !== gi))}>Удалить {g.name}</button>
        </fieldset>)}
      </details>
    </div><div className="qc-decision">
      <h3>Результат модели</h3><p>{modelVerdict(row)}</p>
      <p>Вероятность нарушения: {row.quality_prob}</p>
      {row.decision_reason === 'highest_scoring_criterion' && <p>Тип выбран как наиболее вероятный при положительном общем результате; его собственный порог не достигнут.</p>}
      {row.decision_reason === 'implant_rule' && <p>Применено дополнительное правило обнаружения импланта.</p>}
      {parse<string[]>(row.review_reasons, []).includes('suspected_metal_requires_review') && <p>Яркий участок требует проверки на металл; ошибка ROI автоматически не установлена.</p>}
      {parse<string[]>(row.review_reasons, []).includes('axis_model_measurement_disagreement') && <p>Измерение оси и классификатор расходятся. Итог критерия получен по измеренному углу; проверьте корректность оси.</p>}
      <details className="qc-analysis-details"><summary>Подробности модели, ROI и анатомия</summary>
        <h3>Измерения</h3><dl>{Object.entries(parse<Record<string, number | null>>(row.measurements, {})).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value === null ? 'Недоступно' : value}</dd></div>)}</dl>
        <ul className="qc-criteria">{Object.entries(criteria).map(([key, value]) => <li key={key}><strong>{catalog.criteria[key] || key}</strong><span>Оценка {value.toFixed(3)} · порог {thresholds[key]?.toFixed(3) ?? 'не указан'} · {key === 'spine_axis' && row.decision_version === '3' ? 'итог определяется измеренным углом, а не этим score' : row.violation_codes.split(';').includes(key) ? 'выявлено' : 'не включено в итог'}</span></li>)}</ul>
        {detail && geometry.some(g => g.kind === 'polygon') && <RoiPanel base={base} geometry={geometry} onChange={next => { changeGeometry(next); setOriginal(false); setHighlight([]) }} />}
        {detail && <AnatomyPanel detail={detail} source={API + base + "/original.png"} geometry={geometry} onChange={next => { changeGeometry(next); setOriginal(false); setHighlight([]) }} />}
        <SpecialistPanel result={detail?.assessment?.specialist_qc} />
        <CriteriaPanel codes={allowed} labels={catalog.criteria} scores={criteria} detected={row.violation_codes?.split(';') || []} angle={parse<Record<string, number | null>>(row.measurements, {}).spine_abs_angle_deg ?? null} onShow={keys => { setHighlight(keys); setShowLayers(true); setOriginal(true) }} />
      </details>
      <h3>Решение специалиста</h3><p>Правки сохраняются отдельно от ответа модели.</p>
      <label>Специалист<input value={author} autoComplete="name" onChange={e => { setAuthor(e.target.value); setDirty(true) }} /></label>
      <label>Качество<select value={quality} onChange={e => { setQuality(e.target.value as typeof quality); setDirty(true) }}><option value="0">Нарушений нет</option><option value="1">Есть нарушения</option><option value="unknown">Невозможно оценить</option></select></label>
      {quality === '1' && <fieldset><legend>Типы нарушений</legend>{allowed.map(code => <label key={code}><input type="checkbox" checked={violations.includes(code)} onChange={e => { setViolations(e.target.checked ? [...violations, code] : violations.filter(v => v !== code)); setDirty(true) }} /> {catalog.criteria[code] || code}</label>)}</fieldset>}
      <label>Вставить шаблон<select value="" onChange={e => { if (e.target.value) { setComment((comment + (comment ? '\n' : '') + e.target.value).slice(0, 2000)); setDirty(true) } }}><option value="">Выберите формулировку</option>{['Проверить расположение ROI в отмеченной области.', 'Требуется дополнительная оценка отмеченной области.', 'Сопоставить результат с исходным изображением.', 'Разметка изменена; требуется повторная проверка.'].map(t => <option key={t}>{t}</option>)}</select></label>
      <label>Комментарий<textarea value={comment} maxLength={2000} onChange={e => { setComment(e.target.value); setDirty(true) }} /></label>
      <details><summary>Последующие действия{followups.length ? ` (${followups.length})` : ''}</summary>
      {followups.map((action, index) => <fieldset key={action.id}><legend>Действие {index + 1}</legend>
        <label>Вид действия<select value={action.kind} onChange={e => { setFollowups(followups.map(a => a.id === action.id ? { ...a, kind: e.target.value as Followup['kind'] } : a)); setDirty(true) }}><option value="markup">Проверка разметки</option><option value="reprocess">Повторная обработка</option><option value="second_opinion">Второе мнение</option><option value="other">Другое</option></select></label>
        <label>Вопрос или задача<textarea maxLength={1000} value={action.question} onChange={e => { setFollowups(followups.map(a => a.id === action.id ? { ...a, question: e.target.value } : a)); setDirty(true) }} /></label>
        <label>Результат действия<textarea maxLength={1000} value={action.resolution} onChange={e => { setFollowups(followups.map(a => a.id === action.id ? { ...a, resolution: e.target.value } : a)); setDirty(true) }} /></label>
        <label><input type="checkbox" checked={action.state === 'done'} onChange={e => { setFollowups(followups.map(a => a.id === action.id ? { ...a, state: e.target.checked ? 'done' : 'open' } : a)); setDirty(true) }} /> Выполнено</label>
        <button onClick={() => { setFollowups(followups.filter(a => a.id !== action.id)); setDirty(true) }}>Удалить действие {index + 1}</button>
      </fieldset>)}
      <button disabled={followups.length >= 30} onClick={() => { setFollowups([...followups, { id: crypto.randomUUID(), kind: 'second_opinion', question: '', state: 'open', resolution: '' }]); setDirty(true) }}>Добавить последующее действие</button></details>
      {quality === '1' && !violations.length && <p id="missing-violation-type">Для подтверждения укажите тип нарушения после проверки снимка или выберите «Невозможно оценить». Черновик можно сохранить без типа.</p>}
      <div className="qc-actions"><button disabled={saving || !detail} onClick={() => save('draft')}>Сохранить черновик</button><button className="primary-button" aria-describedby={quality === '1' && !violations.length ? 'missing-violation-type' : undefined} disabled={saving || !detail || (quality === '1' && !violations.length)} onClick={() => save('confirmed')}>Подтвердить решение</button></div>
      <p role="status">{dirty ? draftMessage || 'Есть несохранённые изменения.' : message}</p>
      <details><summary>История решений ({history.length})</summary>{history.map(h => <article key={h.revision}><strong>Версия {h.revision} · {h.author}</strong><a href={`${API}${base}/reviews/${h.revision}/report.html`} download>Скачать отчёт версии {h.revision}</a><button onClick={() => setComparison(h.revision)}>Сравнить с текущей разметкой</button><p>{h.status === 'confirmed' ? 'Подтверждено' : h.status === 'draft' ? 'Черновик' : 'Невозможно оценить'} · {h.comment}</p>{h.measurements && Object.entries(h.measurements).map(([name, m]) => <p key={name}>{name}: {m.length_mm} мм · {m.angle_from_vertical_deg}° от вертикали</p>)}</article>)}</details>
    </div></div></fieldset>
    {detail && comparison !== null && history.find(h => h.revision === comparison) && <section aria-label="Сравнение решений"><h3>Версия {comparison} и текущая рабочая разметка</h3><p>Автор выбранной версии: {history.find(h => h.revision === comparison)?.author}. Общий масштаб; окна прокручиваются независимо.</p><button onClick={() => setComparison(null)}>Закрыть сравнение</button><div className="qc-review-grid"><div><h4>Сохранённая версия {comparison}</h4><p>{history.find(h => h.revision === comparison)?.comment}</p><div className="qc-viewer"><GeometryViewer detail={detail} source={API + base + '/original.png'} geometry={history.find(h => h.revision === comparison)!.geometry} original={false} zoom={zoom} showLayers editable={false} onChange={() => {}} /></div></div><div><h4>Текущая рабочая версия</h4><p>{comment}</p><div className="qc-viewer"><GeometryViewer detail={detail} source={API + base + '/original.png'} geometry={geometry} original={false} zoom={zoom} showLayers editable={false} onChange={() => {}} /></div></div></div></section>}
  </section>
}
