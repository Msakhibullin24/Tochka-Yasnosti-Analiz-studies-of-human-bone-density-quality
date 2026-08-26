import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  CalendarRange,
  Check,
  CheckCircle2,
  ChevronLeft,
  CircleHelp,
  Download,
  Eye,
  EyeOff,
  FileCheck2,
  FileImage,
  GitCompareArrows,
  History,
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
import { DicomAnalysisError, isDicomFile, MAX_DICOM_BYTES } from './services/analysis'
import { analyzeStudy } from './services/ml-analysis'
import type { ComparisonStatus, CriterionStatus, Landmark, Locale, SiteChange, Study, StudyStatus, TrendStatus } from './types'

type Tab = 'analysis' | 'dynamics' | 'dicom'

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
  return <img className="brand-mark" src="/tochka-logo.svg" alt="" aria-hidden="true" />
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
        <radialGradient id="bodyGlow"><stop offset="0" stopColor="#5C6D89" stopOpacity=".62" /><stop offset="1" stopColor="#111E3B" stopOpacity=".08" /></radialGradient>
        <linearGradient id="bone" x1="0" x2="1"><stop stopColor="#8FB0FF" /><stop offset=".5" stopColor="#FFFFFF" /><stop offset="1" stopColor="#5C73A7" /></linearGradient>
        <filter id="soft"><feGaussianBlur stdDeviation="11" /></filter>
        <filter id="boneSoft"><feGaussianBlur stdDeviation="1.2" /></filter>
      </defs>
      <rect width="440" height="560" rx="22" fill="#070E20" />
      <ellipse cx="220" cy="286" rx="160" ry="266" fill="url(#bodyGlow)" filter="url(#soft)" />
      <path d="M76 147 C96 219 99 348 71 465 M364 147 C342 230 341 355 369 465" fill="none" stroke="#8FB0FF" strokeOpacity=".22" strokeWidth="21" filter="url(#soft)" />
      <path d="M111 477 C148 429 179 416 220 449 C261 416 294 430 331 477" fill="none" stroke="#D9E5FA" strokeOpacity=".38" strokeWidth="28" filter="url(#boneSoft)" />
      <path d="M205 53 C192 77 195 100 207 117 M236 53 C248 76 246 99 234 117" fill="none" stroke="#A9C3FF" strokeOpacity=".62" strokeWidth="10" filter="url(#boneSoft)" />
      {vertebrae.map((v, index) => (
        <g key={v.y} filter="url(#boneSoft)">
          <path d={`M${v.x} ${v.y + 13} Q220 ${v.y - 8} ${v.x + v.w} ${v.y + 13} L${v.x + v.w - 7} ${v.y + 57} Q220 ${v.y + 71} ${v.x + 8} ${v.y + 57} Z`} fill="url(#bone)" opacity={0.74 + index * .04} />
          <ellipse cx="220" cy={v.y + 32} rx={v.w * .21} ry="15" fill="#45536D" opacity=".42" />
          <path d={`M220 ${v.y + 53} L220 ${v.y + 75}`} stroke="#D9E5FA" strokeOpacity=".68" strokeWidth="12" strokeLinecap="round" />
        </g>
      ))}
      <path d="M184 417 Q220 398 256 417 L244 474 Q220 496 196 474Z" fill="url(#bone)" opacity=".74" />
      <g opacity=".13" fill="#fff">
        {Array.from({ length: 34 }, (_, i) => <circle key={i} cx={82 + ((i * 53) % 280)} cy={56 + ((i * 79) % 440)} r={1 + (i % 3)} />)}
      </g>
      {overlay && (
        <g className="roi-overlay">
          {vertebrae.map((v) => <rect key={v.y} x={v.x - 11} y={v.y - 8} width={v.w + 22} height="79" rx="8" fill="none" stroke="#A9C3FF" strokeWidth="2" strokeDasharray="6 5" />)}
          <path d="M128 83 C94 181 103 405 142 488 M313 83 C345 190 336 405 299 488" fill="none" stroke="#2456E6" strokeWidth="2" strokeDasharray="7 6" />
          <g transform="translate(284 280)">
            <circle r="13" fill="#0B1D4E" stroke="#A9C3FF" strokeWidth="2" />
            <path d="M-5 0 L-1 5 L7-6" fill="none" stroke="#A9C3FF" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
          </g>
          <text x="290" y="271" fill="#D9E5FA" fontSize="12" fontFamily="Manrope, system-ui">L3</text>
        </g>
      )}
    </svg>
  )
}

function HipImage({ overlay }: { overlay: boolean }) {
  return (
    <svg viewBox="0 0 440 560" className="dexa-svg" role="img" aria-label="Денситометрическое изображение проксимального отдела бедра">
      <defs>
        <radialGradient id="hipGlow"><stop stopColor="#5C6D89" stopOpacity=".48" /><stop offset="1" stopColor="#111E3B" stopOpacity=".05" /></radialGradient>
        <linearGradient id="hipBone" x1="0" x2="1"><stop stopColor="#8FB0FF" /><stop offset=".5" stopColor="#FFFFFF" /><stop offset="1" stopColor="#5C73A7" /></linearGradient>
        <filter id="hipSoft"><feGaussianBlur stdDeviation="1.4" /></filter>
      </defs>
      <rect width="440" height="560" rx="22" fill="#070E20" />
      <ellipse cx="223" cy="288" rx="177" ry="269" fill="url(#hipGlow)" />
      <path d="M99 108 C123 72 200 58 252 91 C288 113 314 162 306 206 C296 263 253 285 207 266 C166 250 126 218 102 183 C86 159 84 130 99 108Z" fill="url(#hipBone)" opacity=".64" filter="url(#hipSoft)" />
      <path d="M118 132 C148 102 198 99 225 119 C193 139 174 171 176 209 C145 192 118 165 118 132Z" fill="#45536D" opacity=".72" />
      <circle cx="277" cy="230" r="54" fill="url(#hipBone)" opacity=".83" filter="url(#hipSoft)" />
      <path d="M247 263 C223 287 214 311 231 336 L275 365 C296 344 312 324 330 316 L348 272 C320 255 295 248 274 249Z" fill="url(#hipBone)" opacity=".82" filter="url(#hipSoft)" />
      <path d="M276 350 C260 402 258 467 273 534 L349 534 C344 457 352 380 334 316Z" fill="url(#hipBone)" opacity=".84" filter="url(#hipSoft)" />
      <path d="M301 365 C290 419 294 487 307 532" fill="none" stroke="#45536D" strokeOpacity=".48" strokeWidth="12" />
      <g opacity=".13" fill="#fff">
        {Array.from({ length: 34 }, (_, i) => <circle key={i} cx={65 + ((i * 61) % 315)} cy={48 + ((i * 83) % 460)} r={1 + (i % 3)} />)}
      </g>
      {overlay && (
        <g>
          <circle cx="277" cy="230" r="65" fill="none" stroke="#A9C3FF" strokeWidth="2" strokeDasharray="7 5" />
          <path d="M218 289 L317 356 L340 313 L249 257Z" fill="#2456E6" fillOpacity=".08" stroke="#2456E6" strokeWidth="2" strokeDasharray="7 5" />
          <rect x="272" y="338" width="68" height="125" rx="8" fill="#A9C3FF" fillOpacity=".06" stroke="#A9C3FF" strokeWidth="2" strokeDasharray="7 5" />
          <path d="M248 280 L265 296" stroke="#D9E5FA" strokeWidth="3" />
          <circle cx="247" cy="279" r="7" fill="#D9E5FA" />
        </g>
      )}
    </svg>
  )
}

const landmarkLinks = [
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_inner_elbow'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_inner_elbow'],
  ['right_shoulder', 'left_shoulder'], ['right_hip_skin', 'left_hip_skin'],
  ['right_hip_skin', 'right_outer_knee'], ['left_hip_skin', 'left_outer_knee'],
] as const

function LandmarkOverlay({ landmarks }: { landmarks: Landmark[] }) {
  const visible = new Map(landmarks.filter((item) => item.visible).map((item) => [item.name, item]))
  return (
    <svg className="landmark-overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">
      <g className="landmark-links">
        {landmarkLinks.map(([from, to]) => {
          const start = visible.get(from)
          const end = visible.get(to)
          return start && end ? <line key={`${from}-${to}`} x1={start.x * 1000} y1={start.y * 1000} x2={end.x * 1000} y2={end.y * 1000} /> : null
        })}
      </g>
      <g className="landmark-points">
        {[...visible.values()].map((item) => <circle key={item.name} cx={item.x * 1000} cy={item.y * 1000} r={item.confidence >= .8 ? 5 : 7} className={item.confidence >= .8 ? '' : 'is-uncertain'} />)}
      </g>
    </svg>
  )
}

function DicomViewer({ study, locale }: { study: Study; locale: Locale }) {
  const c = copy[locale]
  const [overlay, setOverlay] = useState(true)
  const [zoom, setZoom] = useState(1)
  const hasPixelPreview = Boolean(study.previewUrl)
  const hasLandmarks = Boolean(study.landmarks?.some((item) => item.visible))
  const canShowOverlay = !hasPixelPreview || hasLandmarks

  const changeZoom = (amount: number) => setZoom((value) => Math.min(1.6, Math.max(.8, Number((value + amount).toFixed(1)))))

  return (
    <section className="viewer-card" aria-labelledby="image-heading">
      <div className="card-heading viewer-heading">
        <div>
          <p className="eyebrow">{c.image}</p>
          <h2 id="image-heading">{c[study.type]}</h2>
        </div>
        <div className="viewer-tools" role="toolbar" aria-label={locale === 'ru' ? 'Управление изображением' : 'Image controls'}>
          {canShowOverlay && <button className={`icon-button ${overlay ? 'is-selected' : ''}`} onClick={() => setOverlay(!overlay)} aria-pressed={overlay} aria-label={overlay ? c.viewOriginal : c.viewMarkup} title={overlay ? c.viewOriginal : c.viewMarkup}>
            {overlay ? <Eye size={18} aria-hidden="true" /> : <EyeOff size={18} aria-hidden="true" />}
          </button>}
          {canShowOverlay && <span className="tool-divider" aria-hidden="true" />}
          <button className="icon-button" onClick={() => changeZoom(-.1)} aria-label={c.zoomOut} title={c.zoomOut} disabled={zoom <= .8}><Minus size={18} /></button>
          <span className="zoom-value" aria-live="polite">{Math.round(zoom * 100)}%</span>
          <button className="icon-button" onClick={() => changeZoom(.1)} aria-label={c.zoomIn} title={c.zoomIn} disabled={zoom >= 1.6}><Plus size={18} /></button>
          <button className="icon-button" onClick={() => setZoom(1)} aria-label={c.resetView} title={c.resetView}><RotateCcw size={17} /></button>
        </div>
      </div>
      <div className="viewer-surface">
        <div className="scan-canvas" style={{ transform: `scale(${zoom})` }}>
          {study.previewUrl
            ? <><img className="dexa-svg uploaded-preview" src={study.previewUrl} alt={locale === 'ru' ? 'Пиксельное изображение из загруженного DICOM' : 'Pixel image from the uploaded DICOM'} />{overlay && study.landmarks && <LandmarkOverlay landmarks={study.landmarks} />}</>
            : study.type === 'spine' ? <SpineImage overlay={overlay} /> : <HipImage overlay={overlay} />}
        </div>
        <div className="orientation-markers" aria-hidden="true"><span>R</span><span>L</span></div>
        <span className="scan-label">DXA · AP</span>
      </div>
      <div className="viewer-footer">
        {hasPixelPreview ? <span className="legend-item preview-note"><Info size={14} aria-hidden="true" />{hasLandmarks ? c.landmarkPreview : c.pixelPreview}</span> : <>
          <span className="legend-title">{c.roiLegend}</span>
          <span className="legend-item"><i className="legend-swatch swatch-green" />{study.type === 'spine' ? c.roiVertebrae : locale === 'ru' ? 'Области шейки и головки' : 'Neck and head regions'}</span>
          <span className="legend-item"><i className="legend-swatch swatch-blue" />{c.roiTissue}</span>
          {study.status !== 'passed' && <span className="legend-item"><i className="legend-swatch swatch-amber" />{c.roiArtifact}</span>}
        </>}
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
      <div className="analysis-scope">
        <ShieldCheck size={17} aria-hidden="true" />
        <div><strong>{c.analysisScope}</strong><span>{c.mode[study.provenance.mode]} · {study.provenance.modelVersion}</span></div>
      </div>
      {study.provenance.warnings.length > 0 && <details className="provenance-details">
        <summary>{c.analysisWarnings} · {study.provenance.warnings.length}</summary>
        <ul>{study.provenance.warnings.map((warning) => <li key={warning[locale]}>{warning[locale]}</li>)}</ul>
      </details>}
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
          <div><dt>{c.modality}</dt><dd>{study.technical.modality}</dd></div>
          <div><dt>{c.imageSize}</dt><dd>{study.technical.columns && study.technical.rows ? `${study.technical.columns} × ${study.technical.rows} px` : '—'}</dd></div>
          <div><dt>{c.pixelSpacing}</dt><dd>{study.technical.pixelSpacing ?? '—'}</dd></div>
          <div><dt>Photometric interpretation</dt><dd>{study.technical.photometricInterpretation ?? '—'}</dd></div>
          <div><dt>Transfer Syntax UID</dt><dd title={study.technical.transferSyntaxUid ?? undefined}>{study.technical.transferSyntaxUid ?? '—'}</dd></div>
          <div><dt>{c.analysisVersion}</dt><dd>{study.provenance.modelVersion}</dd></div>
          {study.routing && <>
            <div><dt>{locale === 'ru' ? 'Маршрут модели' : 'Model route'}</dt><dd title={study.routing.evidence.join(', ')}>{study.routing.protocol} · {Math.round(study.routing.confidence * 100)}%</dd></div>
            <div><dt>{locale === 'ru' ? 'Статус модели' : 'Model status'}</dt><dd title={study.routing.modelKey}>{study.routing.modelStatus} · {study.routing.source}</dd></div>
          </>}
        </dl>
      </section>
    </div>
  )
}

const localeCode = (locale: Locale) => locale === 'ru' ? 'ru-RU' : 'en-GB'

const formatDate = (value: string, locale: Locale) => new Intl.DateTimeFormat(localeCode(locale), {
  day: '2-digit', month: 'short', year: 'numeric',
}).format(new Date(`${value}T00:00:00Z`))

const formatBmd = (value: number, locale: Locale) => new Intl.NumberFormat(localeCode(locale), {
  minimumFractionDigits: 3, maximumFractionDigits: 3,
}).format(value)

const formatChange = (value: number, locale: Locale) => `${value > 0 ? '+' : ''}${new Intl.NumberFormat(localeCode(locale), {
  minimumFractionDigits: 1, maximumFractionDigits: 1,
}).format(value)}%`

const trendLabel = (status: TrendStatus, locale: Locale) => ({
  'significant-gain': locale === 'ru' ? 'Значимый прирост' : 'Significant gain',
  stable: locale === 'ru' ? 'Без значимых изменений' : 'No significant change',
  'significant-loss': locale === 'ru' ? 'Значимое снижение' : 'Significant loss',
  'not-comparable': locale === 'ru' ? 'Не интерпретируется' : 'Not interpretable',
})[status]

const comparisonLabel = (status: ComparisonStatus, locale: Locale) => ({
  comparable: locale === 'ru' ? 'Исследования сопоставимы' : 'Studies are comparable',
  review: locale === 'ru' ? 'Сопоставимость требует проверки' : 'Comparability needs review',
  'not-comparable': locale === 'ru' ? 'Сравнение заблокировано' : 'Comparison blocked',
})[status]

function TrendChart({ site, baselineStudyId, locale }: { site: SiteChange; baselineStudyId: string; locale: Locale }) {
  const chartWidth = 760
  const chartHeight = 280
  const plot = { left: 58, right: 726, top: 28, bottom: 224 }
  const lscDelta = site.baselineBmd * site.lscPercent / 100
  const values = [...site.history.map((point) => point.bmd), site.baselineBmd - lscDelta, site.baselineBmd + lscDelta]
  const rawMin = Math.min(...values)
  const rawMax = Math.max(...values)
  const padding = Math.max((rawMax - rawMin) * .18, .006)
  const min = rawMin - padding
  const max = rawMax + padding
  const x = (index: number) => plot.left + (index / Math.max(site.history.length - 1, 1)) * (plot.right - plot.left)
  const y = (value: number) => plot.bottom - ((value - min) / (max - min)) * (plot.bottom - plot.top)
  const path = site.history.map((point, index) => `${index === 0 ? 'M' : 'L'} ${x(index)} ${y(point.bmd)}`).join(' ')
  const baselineIndex = Math.max(0, site.history.findIndex((point) => point.studyId === baselineStudyId))
  const labelId = `trend-title-${site.site}`
  const descriptionId = `trend-description-${site.site}`
  const statusClass = site.status === 'significant-gain' ? 'gain' : site.status === 'significant-loss' ? 'loss' : site.status === 'stable' ? 'stable' : 'blocked'

  return (
    <div className={`trend-chart trend-${statusClass}`}>
      <div className="trend-chart-head">
        <div><span>{locale === 'ru' ? 'Динамика BMD' : 'BMD trend'}</span><strong>{site.label[locale]}</strong></div>
        <span className={`trend-result trend-result-${statusClass}`}>{trendLabel(site.status, locale)}</span>
      </div>
      <svg className="trend-svg" viewBox={`0 0 ${chartWidth} ${chartHeight}`} role="img" aria-labelledby={`${labelId} ${descriptionId}`}>
        <title id={labelId}>{locale === 'ru' ? `График BMD: ${site.label.ru}` : `BMD chart: ${site.label.en}`}</title>
        <desc id={descriptionId}>{locale === 'ru'
          ? `Изменение от ${formatBmd(site.baselineBmd, locale)} до ${formatBmd(site.currentBmd, locale)} грамма на квадратный сантиметр. LSC ${site.lscPercent} процента.`
          : `Change from ${formatBmd(site.baselineBmd, locale)} to ${formatBmd(site.currentBmd, locale)} grams per square centimetre. LSC ${site.lscPercent} percent.`}</desc>
        {[0, .5, 1].map((fraction) => {
          const value = max - (max - min) * fraction
          const lineY = plot.top + (plot.bottom - plot.top) * fraction
          return <g key={fraction}><line className="chart-grid-line" x1={plot.left} x2={plot.right} y1={lineY} y2={lineY} /><text className="chart-axis-label" x={plot.left - 10} y={lineY + 4} textAnchor="end">{value.toFixed(2)}</text></g>
        })}
        <rect className="lsc-band" x={x(baselineIndex)} y={y(site.baselineBmd + lscDelta)} width={plot.right - x(baselineIndex)} height={y(site.baselineBmd - lscDelta) - y(site.baselineBmd + lscDelta)} rx="7" />
        <line className="baseline-line" x1={x(baselineIndex)} x2={plot.right} y1={y(site.baselineBmd)} y2={y(site.baselineBmd)} />
        <text className="lsc-label" x={plot.right - 3} y={y(site.baselineBmd + lscDelta) - 8} textAnchor="end">LSC ±{site.lscPercent}%</text>
        <path className="trend-path" d={path} />
        {site.history.map((point, index) => (
          <g key={`${point.studyId}-${point.date}`}>
            <circle className="trend-point" cx={x(index)} cy={y(point.bmd)} r="6" />
            <text className="trend-value" x={x(index)} y={y(point.bmd) - 14} textAnchor="middle">{point.bmd.toFixed(3)}</text>
            <text className="chart-date" x={x(index)} y={plot.bottom + 27} textAnchor="middle">{new Date(`${point.date}T00:00:00Z`).getUTCFullYear()}</text>
          </g>
        ))}
      </svg>
      <div className="trend-legend" aria-hidden="true">
        <span><i className="legend-line-current" />BMD</span>
        <span><i className="legend-band" />{locale === 'ru' ? 'Диапазон LSC учреждения' : 'Facility LSC range'}</span>
      </div>
    </div>
  )
}

function LongitudinalPanel({ study, locale, onUpload }: { study: Study; locale: Locale; onUpload: () => void }) {
  const analysis = study.longitudinal
  const [selectedSite, setSelectedSite] = useState(analysis?.sites[0]?.site)

  if (!analysis) return (
    <section className="longitudinal-empty" aria-labelledby="dynamics-empty-title">
      <span className="empty-trend-icon"><History size={28} aria-hidden="true" /></span>
      <p className="eyebrow">Follow-up DXA</p>
      <h2 id="dynamics-empty-title">{locale === 'ru' ? 'Динамика пока недоступна' : 'Trend analysis is not available yet'}</h2>
      <p>{locale === 'ru'
        ? 'Для сравнения нужно предыдущее качественное исследование той же области и структурированные значения BMD. Система сначала проверит аппарат, протокол, укладку и ROI.'
        : 'A prior quality study of the same region and structured BMD values are required. The system will first check device, protocol, positioning, and ROIs.'}</p>
      <button className="primary-button" onClick={onUpload}><Upload size={17} aria-hidden="true" />{locale === 'ru' ? 'Загрузить предыдущее исследование' : 'Upload prior study'}</button>
    </section>
  )

  const site = analysis.sites.find((item) => item.site === selectedSite) ?? analysis.sites[0]
  const statusClass = analysis.status === 'comparable' ? 'comparable' : analysis.status === 'review' ? 'review' : 'blocked'
  const changeIcon = site.percentChange > 0 ? <ArrowUpRight size={21} aria-hidden="true" /> : <ArrowDownRight size={21} aria-hidden="true" />

  return (
    <div className="longitudinal-layout">
      <section className={`trend-hero comparison-${statusClass}`} aria-labelledby="dynamics-heading">
        <div className="trend-hero-heading">
          <div>
            <p className="eyebrow">Follow-up DXA</p>
            <div className="trend-title-line"><h2 id="dynamics-heading">{locale === 'ru' ? 'Анализ в динамике' : 'Longitudinal analysis'}</h2>{analysis.assumed && <span className="assumed-badge">{locale === 'ru' ? 'Предполагаемый результат' : 'Assumed result'}</span>}</div>
            <p>{locale === 'ru' ? 'Сравнение с исходным исследованием и проверка значимости изменения BMD.' : 'Comparison with baseline and assessment of whether the BMD change is significant.'}</p>
          </div>
          <span className={`comparison-badge comparison-badge-${statusClass}`}>{analysis.status === 'comparable' ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}{comparisonLabel(analysis.status, locale)}</span>
        </div>

        {analysis.sites.length > 1 && <div className="site-switcher" aria-label={locale === 'ru' ? 'Область измерения' : 'Measurement site'}>
          {analysis.sites.map((item) => <button key={item.site} aria-pressed={site.site === item.site} onClick={() => setSelectedSite(item.site)}>{item.label[locale]}</button>)}
        </div>}

        <div className="trend-metrics">
          <article><span className="metric-icon"><CalendarRange size={19} /></span><div><span>{locale === 'ru' ? 'Интервал' : 'Interval'}</span><strong>{analysis.intervalMonths} {locale === 'ru' ? 'мес.' : 'mo'}</strong><small>{formatDate(analysis.baselineDate, locale)} → {formatDate(analysis.currentDate, locale)}</small></div></article>
          <article><span className="metric-icon"><History size={19} /></span><div><span>Baseline</span><strong>{analysis.baselineStudyId}</strong><small>{formatBmd(site.baselineBmd, locale)} g/cm²</small></div></article>
          <article className={`change-metric change-${site.status}`}>{changeIcon}<div><span>ΔBMD</span><strong>{formatChange(site.percentChange, locale)}</strong><small>{site.absoluteChange > 0 ? '+' : ''}{formatBmd(site.absoluteChange, locale)} g/cm²</small></div></article>
          <article><span className="metric-icon"><GitCompareArrows size={19} /></span><div><span>LSC</span><strong>{new Intl.NumberFormat(localeCode(locale), { minimumFractionDigits: 1 }).format(site.lscPercent)}%</strong><small>{locale === 'ru' ? 'порог учреждения' : 'facility threshold'}</small></div></article>
        </div>
      </section>

      <div className="longitudinal-grid">
        <section className="trend-chart-card" aria-label={locale === 'ru' ? 'График и история измерений' : 'Chart and measurement history'}>
          <TrendChart site={site} baselineStudyId={analysis.baselineStudyId} locale={locale} />
          <div className="trend-table-wrap">
            <table className="trend-table">
              <caption>{locale === 'ru' ? 'Точные значения BMD' : 'Exact BMD values'}</caption>
              <thead><tr><th>{locale === 'ru' ? 'Дата' : 'Date'}</th><th>{locale === 'ru' ? 'Исследование' : 'Study'}</th><th>BMD, g/cm²</th></tr></thead>
              <tbody>{site.history.map((point) => <tr key={`${point.studyId}-table`}><td>{formatDate(point.date, locale)}</td><td>{point.studyId}</td><td>{formatBmd(point.bmd, locale)}</td></tr>)}</tbody>
            </table>
          </div>
        </section>

        <aside className="comparability-card" aria-labelledby="comparability-heading">
          <div className="card-heading"><div><p className="eyebrow">Quality gate</p><h2 id="comparability-heading">{locale === 'ru' ? 'Сопоставимость' : 'Comparability'}</h2><p>{locale === 'ru' ? `${analysis.checks.filter((check) => check.passed).length} из ${analysis.checks.length} проверок пройдено` : `${analysis.checks.filter((check) => check.passed).length} of ${analysis.checks.length} checks passed`}</p></div><strong className="comparison-confidence">{analysis.confidence}%</strong></div>
          <div className="comparison-checks">
            {analysis.checks.map((check) => <article className={check.passed ? 'check-passed' : 'check-failed'} key={check.id}>
              <span className="check-icon">{check.passed ? <Check size={17} /> : <X size={17} />}</span>
              <div><h3>{check.label[locale]}{check.critical && <span>{locale === 'ru' ? 'Критично' : 'Critical'}</span>}</h3><p>{check.detail[locale]}</p></div>
            </article>)}
          </div>
          <div className={`trend-decision decision-${statusClass}`}>
            <strong>{locale === 'ru' ? 'Вывод' : 'Decision'}</strong>
            <p>{analysis.summary[locale]}</p>
            <span>{analysis.recommendation[locale]}</span>
          </div>
          <p className="lsc-disclaimer"><Info size={15} />{locale === 'ru' ? 'LSC в демо задан по минимально приемлемой точности ISCD. В клинике нужен собственный LSC аппарата и оператора.' : 'Demo LSC uses ISCD minimum acceptable precision. Production use requires facility-, device-, and operator-specific LSC.'}</p>
        </aside>
      </div>
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
            <input ref={inputRef} className="sr-only" type="file" accept=".dcm,.dicom,application/dicom" aria-label={c.chooseDicom} tabIndex={-1} onChange={(event) => { const file = event.target.files?.[0]; if (file) onFile(file) }} />
            {error && <p className="upload-error" role="alert"><AlertTriangle size={16} />{error}</p>}
            <div className="privacy-note"><ShieldCheck size={18} aria-hidden="true" /><span><strong>{c.localProcessing}</strong>{locale === 'ru' ? 'Файл передаётся только настроенному сервису анализа; прямые идентификаторы не включаются в ответ и логи приложения.' : 'The file is sent only to the configured analysis service; direct identifiers are excluded from its response and application logs.'}</span></div>
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

  const handleTabKey = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const order: Tab[] = ['analysis', 'dynamics', 'dicom']
    const currentIndex = order.indexOf(tab)
    const next = event.key === 'ArrowRight' ? order[(currentIndex + 1) % order.length]
      : event.key === 'ArrowLeft' ? order[(currentIndex - 1 + order.length) % order.length]
        : event.key === 'Home' ? order[0]
          : event.key === 'End' ? order[order.length - 1]
            : undefined
    if (!next) return
    event.preventDefault()
    setTab(next)
    window.requestAnimationFrame(() => document.getElementById(`tab-${next}`)?.focus())
  }

  const handleFile = async (file: File) => {
    if (!isDicomFile(file)) { setUploadError(c.wrongFile); return }
    if (file.size > MAX_DICOM_BYTES) { setUploadError(c.fileTooLarge); return }
    setUploadError('')
    setUploadBusy(true)
    try {
      const study = await analyzeStudy(file)
      setStudies((current) => [study, ...current.filter((item) => item.id !== study.id)])
      setSelectedId(study.id)
      setUploadOpen(false)
      setAnnouncement(c.added)
      window.setTimeout(() => setAnnouncement(''), 4000)
    } catch (error) {
      const code = error instanceof DicomAnalysisError ? error.code : 'invalid-dicom'
      setUploadError(code === 'empty-file' ? c.emptyFile : code === 'file-too-large' ? c.fileTooLarge : c.invalidDicom)
    } finally {
      setUploadBusy(false)
    }
  }

  const exportReport = () => {
    const report = {
      schemaVersion: '1.2', generatedAt: new Date().toISOString(), demo: selected.provenance.mode !== 'validated-model', study: {
        id: selected.id, patientId: selected.patientId, filename: selected.filename,
        status: selected.status, qualityScore: selected.score, confidence: selected.confidence,
        technical: selected.technical, provenance: selected.provenance, privacy: selected.privacy,
        routing: selected.routing ?? null,
        landmarks: selected.landmarks ?? [],
        criteria: selected.criteria.map((item) => ({ code: item.code, name: item.title[locale], status: item.status, confidence: item.confidence, detail: item.detail[locale] })),
        recommendation: selected.recommendation[locale],
        longitudinal: selected.longitudinal ? {
          ...selected.longitudinal,
          summary: selected.longitudinal.summary[locale],
          recommendation: selected.longitudinal.recommendation[locale],
          checks: selected.longitudinal.checks.map((check) => ({ ...check, label: check.label[locale], detail: check.detail[locale] })),
          sites: selected.longitudinal.sites.map((site) => ({ ...site, label: site.label[locale] })),
        } : null,
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
          <div className={`privacy-pill ${selected.privacy.deidentificationVerified ? '' : 'privacy-unverified'}`}>
            {selected.privacy.deidentificationVerified ? <ShieldCheck size={16} aria-hidden="true" /> : <AlertTriangle size={16} aria-hidden="true" />}
            {selected.privacy.deidentificationVerified ? c.anonymized : c.anonymizationUnverified}
          </div>
          <div className="top-actions">
            <button className="language-switch" onClick={() => setLocale(locale === 'ru' ? 'en' : 'ru')} aria-label={locale === 'ru' ? 'Switch to English' : 'Переключить на русский'}>
              <Languages size={16} /><span className={locale === 'ru' ? 'active' : ''}>RU</span><i /> <span className={locale === 'en' ? 'active' : ''}>EN</span>
            </button>
            <button className="icon-button help-button" aria-label={c.help} title={c.help} onClick={() => { setAnnouncement(c.helpMessage); window.setTimeout(() => setAnnouncement(''), 6000) }}><CircleHelp size={19} /></button>
          </div>
        </header>

        <main id="main-content" className="main-content">
          <div id="mobile-studies" className="mobile-study-strip" aria-label={c.allStudies} tabIndex={-1}>
            {studies.slice(0, 4).map((study) => <StudyListItem key={study.id} study={study} locale={locale} active={selected.id === study.id} onClick={() => selectStudy(study.id)} />)}
          </div>
          <section className="study-header" aria-labelledby="page-title">
            <div className="study-title">
              <button className="back-button" aria-label={c.backToList} onClick={() => document.getElementById('mobile-studies')?.focus()}><ChevronLeft size={20} /></button>
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

          <div className="tabs" role="tablist" aria-label={locale === 'ru' ? 'Разделы исследования' : 'Study sections'} onKeyDown={handleTabKey}>
            <button id="tab-analysis" role="tab" aria-selected={tab === 'analysis'} aria-controls="panel-analysis" tabIndex={tab === 'analysis' ? 0 : -1} className={tab === 'analysis' ? 'is-active' : ''} onClick={() => setTab('analysis')}>{c.analysis}</button>
            <button id="tab-dynamics" role="tab" aria-selected={tab === 'dynamics'} aria-controls="panel-dynamics" tabIndex={tab === 'dynamics' ? 0 : -1} className={tab === 'dynamics' ? 'is-active' : ''} onClick={() => setTab('dynamics')}>{c.dynamics}</button>
            <button id="tab-dicom" role="tab" aria-selected={tab === 'dicom'} aria-controls="panel-dicom" tabIndex={tab === 'dicom' ? 0 : -1} className={tab === 'dicom' ? 'is-active' : ''} onClick={() => setTab('dicom')}>{c.dicom}</button>
          </div>

          {tab === 'analysis' && <div id="panel-analysis" role="tabpanel" aria-labelledby="tab-analysis" className="analysis-grid"><DicomViewer key={selected.id} study={selected} locale={locale} /><CriteriaPanel study={selected} locale={locale} /></div>}
          {tab === 'dynamics' && <div id="panel-dynamics" role="tabpanel" aria-labelledby="tab-dynamics"><LongitudinalPanel key={selected.id} study={selected} locale={locale} onUpload={() => { setUploadError(''); setUploadOpen(true) }} /></div>}
          {tab === 'dicom' && <div id="panel-dicom" role="tabpanel" aria-labelledby="tab-dicom"><MetadataPanel study={selected} locale={locale} /></div>}
        </main>
      </div>
      <UploadDialog open={uploadOpen} locale={locale} busy={uploadBusy} error={uploadError} onClose={() => setUploadOpen(false)} onFile={handleFile} />
      <div className="sr-only" role="status" aria-live="polite">{announcement}</div>
      {announcement && <div className="toast"><CheckCircle2 size={18} />{announcement}</div>}
    </div>
  )
}

export default App
