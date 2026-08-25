import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronLeft,
  CircleHelp,
  Download,
  Eye,
  EyeOff,
  FileCheck2,
  FileImage,
  Info,
  Languages,
  LoaderCircle,
  Maximize2,
  Minus,
  Plus,
  RotateCcw,
  Search,
  ShieldCheck,
  Upload,
  X,
  XCircle,
} from 'lucide-react'
import { copy, initialStudies } from './data'
import { analyzeDicom, isDicomFile, MAX_DICOM_BYTES } from './services/analysis'
import type { CriterionStatus, Locale, Study, StudyStatus } from './types'

type Tab = 'analysis' | 'dicom'

const statusIcon = (status: StudyStatus | CriterionStatus, size = 16) => {
  if (status === 'passed' || status === 'pass') return <CheckCircle2 size={size} aria-hidden="true" />
  if (status === 'review' || status === 'warning') return <AlertTriangle size={size} aria-hidden="true" />
  return <XCircle size={size} aria-hidden="true" />
}

function StatusBadge({ status, locale }: { status: StudyStatus; locale: Locale }) {
  const c = copy[locale]
  const label = status === 'passed' ? c.passed : status === 'review' ? c.review : c.rejected
  return <span className={`status-badge status-${status}`}>{statusIcon(status)}{label}</span>
}

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <span />
      <span />
      <span />
    </span>
  )
}

function StudyListItem({ study, locale, active, onClick }: { study: Study; locale: Locale; active: boolean; onClick: () => void }) {
  const c = copy[locale]
  return (
    <button className={`study-row ${active ? 'is-active' : ''}`} onClick={onClick} aria-current={active ? 'true' : undefined}>
      <span className={`study-status-dot dot-${study.status}`} aria-hidden="true" />
      <span className="study-row-content">
        <span className="study-row-top">
          <strong>{study.patientId}</strong>
          <span className={`study-mini-status text-${study.status}`}>{study.status === 'passed' ? c.passed : study.status === 'review' ? c.review : c.rejected}</span>
        </span>
        <span className="study-row-meta">{c[study.type]} · {study.acquiredAt.split(' · ')[0]}</span>
      </span>
    </button>
  )
}

function SpineImage({ overlay }: { overlay: boolean }) {
  const vertebrae = [
    { y: 118, w: 84, x: 178 }, { y: 190, w: 92, x: 174 }, { y: 264, w: 99, x: 170 }, { y: 340, w: 106, x: 167 },
  ]
  return (
    <svg viewBox="0 0 440 560" className="dexa-svg" role="img" aria-label="Денситометрическое изображение поясничного отдела позвоночника">
      <defs>
        <radialGradient id="bodyGlow"><stop offset="0" stopColor="#8c8f8f" stopOpacity=".62" /><stop offset="1" stopColor="#171a1b" stopOpacity=".08" /></radialGradient>
        <linearGradient id="bone" x1="0" x2="1"><stop stopColor="#c5c8c5" /><stop offset=".5" stopColor="#f2f1e9" /><stop offset="1" stopColor="#a2a7a4" /></linearGradient>
        <filter id="soft"><feGaussianBlur stdDeviation="11" /></filter>
        <filter id="boneSoft"><feGaussianBlur stdDeviation="1.2" /></filter>
      </defs>
      <rect width="440" height="560" rx="8" fill="#0b0d0e" />
      <ellipse cx="220" cy="286" rx="160" ry="266" fill="url(#bodyGlow)" filter="url(#soft)" />
      <path d="M76 147 C96 219 99 348 71 465 M364 147 C342 230 341 355 369 465" fill="none" stroke="#a4aaa8" strokeOpacity=".22" strokeWidth="21" filter="url(#soft)" />
      <path d="M111 477 C148 429 179 416 220 449 C261 416 294 430 331 477" fill="none" stroke="#d1d4d0" strokeOpacity=".38" strokeWidth="28" filter="url(#boneSoft)" />
      <path d="M205 53 C192 77 195 100 207 117 M236 53 C248 76 246 99 234 117" fill="none" stroke="#c8cbc8" strokeOpacity=".62" strokeWidth="10" filter="url(#boneSoft)" />
      {vertebrae.map((v, index) => (
        <g key={v.y} filter="url(#boneSoft)">
          <path d={`M${v.x} ${v.y + 13} Q220 ${v.y - 8} ${v.x + v.w} ${v.y + 13} L${v.x + v.w - 7} ${v.y + 57} Q220 ${v.y + 71} ${v.x + 8} ${v.y + 57} Z`} fill="url(#bone)" opacity={0.74 + index * .04} />
          <ellipse cx="220" cy={v.y + 32} rx={v.w * .21} ry="15" fill="#6f7473" opacity=".42" />
          <path d={`M220 ${v.y + 53} L220 ${v.y + 75}`} stroke="#dedfd8" strokeOpacity=".68" strokeWidth="12" strokeLinecap="round" />
        </g>
      ))}
      <path d="M184 417 Q220 398 256 417 L244 474 Q220 496 196 474Z" fill="url(#bone)" opacity=".74" />
      <g opacity=".13" fill="#fff">
        {Array.from({ length: 34 }, (_, i) => <circle key={i} cx={82 + ((i * 53) % 280)} cy={56 + ((i * 79) % 440)} r={1 + (i % 3)} />)}
      </g>
      {overlay && (
        <g className="roi-overlay">
          {vertebrae.map((v, index) => <rect key={v.y} x={v.x - 11} y={v.y - 8} width={v.w + 22} height="79" rx="8" fill="none" stroke="#6ee7b7" strokeWidth="2" strokeDasharray="6 5" />)}
          <path d="M128 83 C94 181 103 405 142 488 M313 83 C345 190 336 405 299 488" fill="none" stroke="#5bc9e8" strokeWidth="2" strokeDasharray="7 6" />
          <g transform="translate(284 280)">
            <circle r="13" fill="#101918" stroke="#6ee7b7" strokeWidth="2" />
            <path d="M-5 0 L-1 5 L7-6" fill="none" stroke="#6ee7b7" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
          </g>
          <text x="290" y="271" fill="#b8f6df" fontSize="12" fontFamily="system-ui">L3</text>
        </g>
      )}
    </svg>
  )
}

function HipImage({ overlay }: { overlay: boolean }) {
  return (
    <svg viewBox="0 0 440 560" className="dexa-svg" role="img" aria-label="Денситометрическое изображение проксимального отдела бедра">
      <defs>
        <radialGradient id="hipGlow"><stop stopColor="#979b99" stopOpacity=".48" /><stop offset="1" stopColor="#171a1b" stopOpacity=".05" /></radialGradient>
        <linearGradient id="hipBone" x1="0" x2="1"><stop stopColor="#a6aaa8" /><stop offset=".5" stopColor="#eeeee7" /><stop offset="1" stopColor="#989d9b" /></linearGradient>
        <filter id="hipSoft"><feGaussianBlur stdDeviation="1.4" /></filter>
      </defs>
      <rect width="440" height="560" rx="8" fill="#0b0d0e" />
      <ellipse cx="223" cy="288" rx="177" ry="269" fill="url(#hipGlow)" />
      <path d="M99 108 C123 72 200 58 252 91 C288 113 314 162 306 206 C296 263 253 285 207 266 C166 250 126 218 102 183 C86 159 84 130 99 108Z" fill="url(#hipBone)" opacity=".64" filter="url(#hipSoft)" />
      <path d="M118 132 C148 102 198 99 225 119 C193 139 174 171 176 209 C145 192 118 165 118 132Z" fill="#303435" opacity=".72" />
      <circle cx="277" cy="230" r="54" fill="url(#hipBone)" opacity=".83" filter="url(#hipSoft)" />
      <path d="M247 263 C223 287 214 311 231 336 L275 365 C296 344 312 324 330 316 L348 272 C320 255 295 248 274 249Z" fill="url(#hipBone)" opacity=".82" filter="url(#hipSoft)" />
      <path d="M276 350 C260 402 258 467 273 534 L349 534 C344 457 352 380 334 316Z" fill="url(#hipBone)" opacity=".84" filter="url(#hipSoft)" />
      <path d="M301 365 C290 419 294 487 307 532" fill="none" stroke="#737876" strokeOpacity=".48" strokeWidth="12" />
      <g opacity=".13" fill="#fff">
        {Array.from({ length: 34 }, (_, i) => <circle key={i} cx={65 + ((i * 61) % 315)} cy={48 + ((i * 83) % 460)} r={1 + (i % 3)} />)}
      </g>
      {overlay && (
        <g>
          <circle cx="277" cy="230" r="65" fill="none" stroke="#6ee7b7" strokeWidth="2" strokeDasharray="7 5" />
          <path d="M218 289 L317 356 L340 313 L249 257Z" fill="#5bc9e8" fillOpacity=".08" stroke="#5bc9e8" strokeWidth="2" strokeDasharray="7 5" />
          <rect x="272" y="338" width="68" height="125" rx="8" fill="#6ee7b7" fillOpacity=".06" stroke="#6ee7b7" strokeWidth="2" strokeDasharray="7 5" />
          <path d="M248 280 L265 296" stroke="#f2bd5b" strokeWidth="3" />
          <circle cx="247" cy="279" r="7" fill="#f2bd5b" />
        </g>
      )}
    </svg>
  )
}

function DicomViewer({ study, locale }: { study: Study; locale: Locale }) {
  const c = copy[locale]
  const [overlay, setOverlay] = useState(true)
  const [zoom, setZoom] = useState(1)

  const changeZoom = (amount: number) => setZoom((value) => Math.min(1.6, Math.max(.8, Number((value + amount).toFixed(1)))))

  return (
    <section className="viewer-card" aria-labelledby="image-heading">
      <div className="card-heading viewer-heading">
        <div>
          <p className="eyebrow">{c.image}</p>
          <h2 id="image-heading">{c[study.type]}</h2>
        </div>
        <div className="viewer-tools" role="toolbar" aria-label={locale === 'ru' ? 'Управление изображением' : 'Image controls'}>
          <button className={`icon-button ${overlay ? 'is-selected' : ''}`} onClick={() => setOverlay(!overlay)} aria-pressed={overlay} aria-label={overlay ? c.viewOriginal : c.viewMarkup} title={overlay ? c.viewOriginal : c.viewMarkup}>
            {overlay ? <Eye size={18} /> : <EyeOff size={18} />}
          </button>
          <span className="tool-divider" aria-hidden="true" />
          <button className="icon-button" onClick={() => changeZoom(-.1)} aria-label={c.zoomOut} title={c.zoomOut} disabled={zoom <= .8}><Minus size={18} /></button>
          <span className="zoom-value" aria-live="polite">{Math.round(zoom * 100)}%</span>
          <button className="icon-button" onClick={() => changeZoom(.1)} aria-label={c.zoomIn} title={c.zoomIn} disabled={zoom >= 1.6}><Plus size={18} /></button>
          <button className="icon-button" onClick={() => setZoom(1)} aria-label={c.resetView} title={c.resetView}><RotateCcw size={17} /></button>
        </div>
      </div>
      <div className="viewer-surface">
        <div className="scan-canvas" style={{ transform: `scale(${zoom})` }}>
          {study.type === 'spine' ? <SpineImage overlay={overlay} /> : <HipImage overlay={overlay} />}
        </div>
        <div className="orientation-markers" aria-hidden="true"><span>R</span><span>L</span></div>
        <span className="scan-label">DXA · AP</span>
      </div>
      <div className="viewer-footer">
        <span className="legend-title">{c.roiLegend}</span>
        <span className="legend-item"><i className="legend-swatch swatch-green" />{study.type === 'spine' ? c.roiVertebrae : locale === 'ru' ? 'Области шейки и головки' : 'Neck and head regions'}</span>
        <span className="legend-item"><i className="legend-swatch swatch-blue" />{c.roiTissue}</span>
        {study.status !== 'passed' && <span className="legend-item"><i className="legend-swatch swatch-amber" />{c.roiArtifact}</span>}
      </div>
    </section>
  )
}

function ScoreRing({ value, label, status }: { value: number; label: string; status: StudyStatus }) {
  return (
    <div className={`score-ring score-${status}`} style={{ '--score': `${value * 3.6}deg` } as React.CSSProperties}>
      <div><strong>{value}</strong><span>/100</span></div>
      <span className="sr-only">{label}: {value} из 100</span>
    </div>
  )
}

function CriteriaPanel({ study, locale }: { study: Study; locale: Locale }) {
  const c = copy[locale]
  const passed = study.criteria.filter((item) => item.status === 'pass').length
  const issues = study.criteria.length - passed
  return (
    <section className="criteria-card" aria-labelledby="criteria-heading">
      <div className="card-heading criteria-heading">
        <div>
          <p className="eyebrow">AI QC</p>
          <h2 id="criteria-heading">{c.criteria}</h2>
          <p>{c.criteriaSummary}</p>
        </div>
        <div className="criteria-counts" aria-label={`${passed} ${c.passedCount}, ${issues} ${c.violationsCount}`}>
          <span className="count-pass"><Check size={15} />{passed}</span>
          {issues > 0 && <span className="count-issue"><AlertTriangle size={15} />{issues}</span>}
        </div>
      </div>
      <div className="criteria-list">
        {study.criteria.map((criterion) => (
          <article className={`criterion criterion-${criterion.status}`} key={criterion.id}>
            <span className="criterion-icon">{statusIcon(criterion.status, 19)}</span>
            <div className="criterion-body">
              <div className="criterion-title-row">
                <h3>{criterion.title[locale]}</h3>
                <span>{criterion.confidence}% {c.confidence}</span>
              </div>
              <p>{criterion.detail[locale]}</p>
            </div>
          </article>
        ))}
      </div>
      <div className={`recommendation recommendation-${study.status}`}>
        <Info size={19} aria-hidden="true" />
        <div><strong>{c.recommendation}</strong><p>{study.recommendation[locale]}</p></div>
      </div>
    </section>
  )
}

function MetadataPanel({ study, locale }: { study: Study; locale: Locale }) {
  const c = copy[locale]
  const fields = [
    [c.patientId, study.patientId], [c.accession, study.accessionNumber], [c.device, study.device],
    [c.institution, study.institution[locale]], [c.operator, study.operator], [c.seriesUid, study.seriesUid],
  ]
  return (
    <div className="metadata-layout">
      <section className="metadata-card" aria-labelledby="metadata-heading">
        <div className="card-heading"><div><p className="eyebrow">DICOM</p><h2 id="metadata-heading">{c.dicomMetadata}</h2></div></div>
        <dl className="metadata-list">
          {fields.map(([label, value]) => <div key={label}><dt>{label}</dt><dd title={value}>{value}</dd></div>)}
        </dl>
      </section>
      <section className="metadata-card" aria-labelledby="technical-heading">
        <div className="card-heading"><div><p className="eyebrow">{c.image}</p><h2 id="technical-heading">{c.technical}</h2></div></div>
        <dl className="metadata-list compact">
          <div><dt>{c.modality}</dt><dd>DXA</dd></div>
          <div><dt>{c.imageSize}</dt><dd>440 × 560 px</dd></div>
          <div><dt>{c.pixelSpacing}</dt><dd>0.93 × 0.93 mm</dd></div>
          <div><dt>Photometric interpretation</dt><dd>MONOCHROME2</dd></div>
        </dl>
      </section>
    </div>
  )
}

function UploadDialog({ open, locale, busy, error, onClose, onFile }: { open: boolean; locale: Locale; busy: boolean; error: string; onClose: () => void; onFile: (file: File) => void }) {
  const c = copy[locale]
  const inputRef = useRef<HTMLInputElement>(null)
  const modalRef = useRef<HTMLElement>(null)
  const busyRef = useRef(busy)
  const closeRef = useRef(onClose)
  const [dragging, setDragging] = useState(false)

  useEffect(() => { busyRef.current = busy }, [busy])
  useEffect(() => { closeRef.current = onClose }, [onClose])

  useEffect(() => {
    if (!open) return
    const previousFocus = document.activeElement as HTMLElement | null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    window.requestAnimationFrame(() => modalRef.current?.querySelector<HTMLElement>('button:not(:disabled)')?.focus())

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !busyRef.current) { event.preventDefault(); closeRef.current(); return }
      if (event.key !== 'Tab' || !modalRef.current) return
      const focusable = Array.from(modalRef.current.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), [tabindex]:not([tabindex="-1"])'))
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.body.style.overflow = previousOverflow
      document.removeEventListener('keydown', handleKeyDown)
      previousFocus?.focus()
    }
  }, [open])

  if (!open) return null

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target && !busy) onClose() }}>
      <section ref={modalRef} className="upload-modal" role="dialog" aria-modal="true" aria-labelledby="upload-title">
        <div className="modal-heading">
          <div><span className="modal-icon"><Upload size={21} /></span><div><h2 id="upload-title">{c.uploadAnother}</h2><p>DICOM · .dcm, .dicom</p></div></div>
          <button className="icon-button" onClick={onClose} aria-label={c.close} disabled={busy}><X size={19} /></button>
        </div>
        {busy ? (
          <div className="processing-state" role="status">
            <LoaderCircle size={34} className="spinner" />
            <h3>{c.processing}</h3>
            <p>{c.processingHint}</p>
            <div className="processing-line"><span /></div>
          </div>
        ) : (
          <>
            <button
              className={`dropzone ${dragging ? 'is-dragging' : ''}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => { event.preventDefault(); setDragging(false); const file = event.dataTransfer.files[0]; if (file) onFile(file) }}
            >
              <span className="dropzone-icon"><FileImage size={29} /></span>
              <strong>{c.dropTitle}</strong>
              <span>{c.dropHint}</span>
            </button>
            <input ref={inputRef} className="sr-only" type="file" accept=".dcm,.dicom,application/dicom" onChange={(event) => { const file = event.target.files?.[0]; if (file) onFile(file) }} />
            {error && <p className="upload-error" role="alert"><AlertTriangle size={16} />{error}</p>}
            <div className="privacy-note"><ShieldCheck size={18} /><span><strong>{c.anonymized}</strong>{locale === 'ru' ? 'Файл обрабатывается только в рамках этой демонстрационной сессии.' : 'The file is processed only within this demo session.'}</span></div>
          </>
        )}
      </section>
    </div>
  )
}

function App() {
  const [locale, setLocale] = useState<Locale>('ru')
  const [studies, setStudies] = useState(initialStudies)
  const [selectedId, setSelectedId] = useState(initialStudies[0].id)
  const [query, setQuery] = useState('')
  const [tab, setTab] = useState<Tab>('analysis')
  const [uploadOpen, setUploadOpen] = useState(false)
  const [uploadBusy, setUploadBusy] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [announcement, setAnnouncement] = useState('')
  const c = copy[locale]
  const selected = studies.find((study) => study.id === selectedId) ?? studies[0]
  const filteredStudies = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return studies
    return studies.filter((study) => `${study.id} ${study.patientId} ${study.filename}`.toLowerCase().includes(normalized))
  }, [query, studies])

  useEffect(() => {
    document.documentElement.lang = locale
    document.title = locale === 'ru' ? 'Osseo AI — контроль качества денситометрии' : 'Osseo AI — densitometry quality control'
  }, [locale])

  const selectStudy = (id: string) => { setSelectedId(id); setTab('analysis') }

  const handleFile = async (file: File) => {
    if (!isDicomFile(file)) { setUploadError(c.wrongFile); return }
    if (file.size > MAX_DICOM_BYTES) { setUploadError(c.fileTooLarge); return }
    setUploadError('')
    setUploadBusy(true)
    const study = await analyzeDicom(file)
    setStudies((current) => [study, ...current])
    setSelectedId(study.id)
    setUploadBusy(false)
    setUploadOpen(false)
    setAnnouncement(c.added)
    window.setTimeout(() => setAnnouncement(''), 4000)
  }

  const exportReport = () => {
    const report = {
      generatedAt: new Date().toISOString(), demo: true, study: {
        id: selected.id, patientId: selected.patientId, filename: selected.filename,
        status: selected.status, qualityScore: selected.score, confidence: selected.confidence,
        criteria: selected.criteria.map((item) => ({ name: item.title[locale], status: item.status, confidence: item.confidence, detail: item.detail[locale] })),
        recommendation: selected.recommendation[locale],
      },
    }
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const href = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = href
    link.download = `osseo-report-${selected.id}.json`
    link.click()
    URL.revokeObjectURL(href)
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">{locale === 'ru' ? 'Перейти к содержимому' : 'Skip to content'}</a>
      <aside className="sidebar" aria-label={c.studies} inert={uploadOpen ? true : undefined}>
        <div className="brand"><BrandMark /><span><strong>{c.product}</strong><small>{c.descriptor}</small></span></div>
        <button className="primary-button sidebar-upload" onClick={() => { setUploadError(''); setUploadOpen(true) }}><Upload size={18} />{c.upload}</button>
        <div className="sidebar-section-heading"><h2>{c.studies}</h2><span>{studies.length}</span></div>
        <label className="search-field"><Search size={17} aria-hidden="true" /><span className="sr-only">{c.search}</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={c.search} /></label>
        <nav className="study-list" aria-label={c.allStudies}>
          {filteredStudies.map((study) => <StudyListItem key={study.id} study={study} locale={locale} active={selected.id === study.id} onClick={() => selectStudy(study.id)} />)}
          {filteredStudies.length === 0 && <div className="empty-list"><Search size={24} /><strong>{c.noMatches}</strong><button onClick={() => setQuery('')}>{c.clearSearch}</button></div>}
        </nav>
        <div className="sidebar-footer">
          <div className="demo-label"><span className="pulse-dot" />{c.demo}</div>
          <p>{c.demoHint}</p>
        </div>
      </aside>

      <div className="workspace" inert={uploadOpen ? true : undefined}>
        <header className="topbar">
          <div className="mobile-brand"><BrandMark /><strong>{c.product}</strong></div>
          <div className="privacy-pill"><ShieldCheck size={16} />{c.anonymized}</div>
          <div className="top-actions">
            <button className="language-switch" onClick={() => setLocale(locale === 'ru' ? 'en' : 'ru')} aria-label={locale === 'ru' ? 'Switch to English' : 'Переключить на русский'}>
              <Languages size={16} /><span className={locale === 'ru' ? 'active' : ''}>RU</span><i /> <span className={locale === 'en' ? 'active' : ''}>EN</span>
            </button>
            <button className="icon-button help-button" aria-label={c.help} title={c.help}><CircleHelp size={19} /></button>
          </div>
        </header>

        <main id="main-content" className="main-content">
          <div className="mobile-study-strip" aria-label={c.allStudies}>
            {studies.slice(0, 4).map((study) => <StudyListItem key={study.id} study={study} locale={locale} active={selected.id === study.id} onClick={() => selectStudy(study.id)} />)}
          </div>
          <section className="study-header" aria-labelledby="page-title">
            <div className="study-title">
              <button className="back-button" aria-label={c.backToList}><ChevronLeft size={20} /></button>
              <div>
                <div className="study-kicker"><span>{selected.id}</span><span aria-hidden="true">•</span><span>{selected.filename}</span></div>
                <h1 id="page-title">{c.qualityAssessment}</h1>
                <p>{selected.patientId} · {c[ selected.type ]} · {selected.acquiredAt}</p>
              </div>
            </div>
            <div className="header-actions">
              <button className="secondary-button" onClick={exportReport}><Download size={17} />{c.export}</button>
              <button className="primary-button" onClick={() => { setUploadError(''); setUploadOpen(true) }}><Upload size={17} />{c.uploadAnother}</button>
            </div>
          </section>

          <section className="summary-grid" aria-label={locale === 'ru' ? 'Сводка анализа' : 'Analysis summary'}>
            <article className="summary-card score-summary">
              <ScoreRing value={selected.score} label={c.qualityScore} status={selected.status} />
              <div><span>{c.qualityScore}</span><StatusBadge status={selected.status} locale={locale} /></div>
            </article>
            <article className="summary-card"><span className="summary-icon icon-mint"><Activity size={19} /></span><div><span>{c.modelConfidence}</span><strong className="tabular">{selected.confidence}%</strong></div></article>
            <article className="summary-card"><span className="summary-icon icon-blue"><Maximize2 size={19} /></span><div><span>{c.studyType}</span><strong>{c[selected.type]}</strong></div></article>
            <article className="summary-card"><span className="summary-icon icon-neutral"><FileCheck2 size={19} /></span><div><span>{c.acquired}</span><strong>{selected.acquiredAt}</strong></div></article>
          </section>

          <div className="tabs" role="tablist" aria-label={locale === 'ru' ? 'Разделы исследования' : 'Study sections'}>
            <button role="tab" aria-selected={tab === 'analysis'} className={tab === 'analysis' ? 'is-active' : ''} onClick={() => setTab('analysis')}>{c.analysis}</button>
            <button role="tab" aria-selected={tab === 'dicom'} className={tab === 'dicom' ? 'is-active' : ''} onClick={() => setTab('dicom')}>{c.dicom}</button>
          </div>

          {tab === 'analysis' ? <div className="analysis-grid"><DicomViewer key={selected.id} study={selected} locale={locale} /><CriteriaPanel study={selected} locale={locale} /></div> : <MetadataPanel study={selected} locale={locale} />}
        </main>
      </div>
      <UploadDialog open={uploadOpen} locale={locale} busy={uploadBusy} error={uploadError} onClose={() => setUploadOpen(false)} onFile={handleFile} />
      <div className="sr-only" role="status" aria-live="polite">{announcement}</div>
      {announcement && <div className="toast"><CheckCircle2 size={18} />{announcement}</div>}
    </div>
  )
}

export default App
