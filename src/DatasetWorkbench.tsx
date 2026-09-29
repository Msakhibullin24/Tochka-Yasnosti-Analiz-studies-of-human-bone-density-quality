import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  Archive,
  Check,
  CheckCircle2,
  Database,
  Download,
  Image as ImageIcon,
  LoaderCircle,
  MousePointer2,
  RefreshCw,
  Save,
  Search,
  ShieldAlert,
  SlidersHorizontal,
  UsersRound,
  Waypoints,
  X,
} from 'lucide-react'
import {
  DatasetApiError,
  getDatasetStudies,
  getDatasetStudy,
  getDatasetSummary,
  getExclusionRegistry,
  getReadinessReport,
  datasetExportUrl,
  resolveDatasetUrl,
  saveDatasetAnnotation,
  type AnnotationAction,
  type AnnotationDefect,
  type AnnotationDocument,
  type AnnotationLandmark,
  type AnnotationRegion,
  type DatasetProtocol,
  type DatasetStudy,
  type DatasetSummary,
  type ReadinessReport,
} from './services/dataset'
import type { Locale } from './types'
import DatasetLongitudinal from './DatasetLongitudinal'

const tx = (locale: Locale, ru: string, en: string) => locale === 'ru' ? ru : en

const protocolLabels: Record<DatasetProtocol, { ru: string; en: string }> = {
  spine_pa: { ru: 'Поясничный отдел', en: 'Lumbar spine' },
  hip_left: { ru: 'Левое бедро', en: 'Left hip' },
  hip_right: { ru: 'Правое бедро', en: 'Right hip' },
  forearm_left: { ru: 'Левое предплечье', en: 'Left forearm' },
  forearm_right: { ru: 'Правое предплечье', en: 'Right forearm' },
  total_body: { ru: 'Всё тело', en: 'Total body' },
  unsupported: { ru: 'Не определён', en: 'Unsupported' },
}

const readinessGateEnglish: Record<string, { label: string; action: string }> = {
  dataset_integrity: { label: 'Dataset integrity', action: 'Resolve missing assets, split violations, and duplicates.' },
  patient_leakage: { label: 'Patient-level leakage', action: 'Rebuild all splits exclusively by patientGroupId.' },
  annotation_coverage: { label: 'Expert annotation', action: 'Annotate at least 80% of the pilot set using the approved labelbook.' },
  double_read: { label: 'Independent double read', action: 'Assign an independent second read to at least 20% of studies.' },
  adjudication: { label: 'Disagreement adjudication', action: 'Resolve every reader disagreement with a third expert.' },
  technical_review: { label: 'Technical flag review', action: 'Annotate every study flagged by the technical baseline.' },
  spine_model: { label: 'PA spine model', action: 'Train, calibrate, and freeze the spine model on a patient-level holdout.' },
  hip_model: { label: 'Hip model', action: 'Train and independently validate the hip model.' },
  audit_chain: { label: 'Audit trail integrity', action: 'Run and verify the pilot save and export scenarios.' },
  audit_key: { label: 'Audit pseudonymization key', action: 'Configure a separate OSSEO_AUDIT_HMAC_KEY of at least 16 bytes in secret storage.' },
  access_control: { label: 'Access control', action: 'Connect OIDC/reverse-proxy RBAC before clinical data are used.' },
  clinical_validation: { label: 'Independent clinical validation', action: 'Complete the silent-mode study and register a CVR-… report ID.' },
  raw_semantics: { label: 'Six-channel raw physics', action: 'Confirm channel order on a phantom or vendor reference before BMD use.' },
}

const defectCatalog: Array<{ code: string; ru: string; en: string; protocols?: DatasetProtocol[] }> = [
  { code: 'wrong_protocol', ru: 'Неверный протокол', en: 'Wrong protocol' },
  { code: 'spine_off_center', ru: 'Смещение позвоночника от центра', en: 'Spine off center', protocols: ['spine_pa'] },
  { code: 'spine_tilt', ru: 'Наклон позвоночника', en: 'Spine tilt', protocols: ['spine_pa'] },
  { code: 'spine_rotation', ru: 'Ротация позвоночника', en: 'Spine rotation', protocols: ['spine_pa'] },
  { code: 'spine_incomplete_coverage', ru: 'Неполный охват L1–L4', en: 'Incomplete L1–L4 coverage', protocols: ['spine_pa'] },
  { code: 'vertebra_numbering', ru: 'Ошибка нумерации', en: 'Vertebra numbering', protocols: ['spine_pa'] },
  { code: 'vertebra_roi', ru: 'Некорректный ROI позвонка', en: 'Incorrect vertebra ROI', protocols: ['spine_pa'] },
  { code: 'vertebra_exclusion', ru: 'Некорректное исключение позвонка', en: 'Incorrect vertebra exclusion', protocols: ['spine_pa'] },
  { code: 'hip_rotation', ru: 'Некорректная ротация бедра', en: 'Incorrect hip rotation', protocols: ['hip_left', 'hip_right'] },
  { code: 'hip_abduction_adduction', ru: 'Отведение или приведение бедра', en: 'Hip abduction or adduction', protocols: ['hip_left', 'hip_right'] },
  { code: 'femur_axis', ru: 'Некорректная ось бедренной кости', en: 'Incorrect femur axis', protocols: ['hip_left', 'hip_right'] },
  { code: 'hip_incomplete_coverage', ru: 'Неполный охват бедра', en: 'Incomplete hip coverage', protocols: ['hip_left', 'hip_right'] },
  { code: 'femoral_neck_roi', ru: 'Некорректный ROI шейки бедра', en: 'Incorrect femoral neck ROI', protocols: ['hip_left', 'hip_right'] },
  { code: 'total_hip_roi', ru: 'Некорректный Total Hip ROI', en: 'Incorrect total hip ROI', protocols: ['hip_left', 'hip_right'] },
  { code: 'body_off_center', ru: 'Тело смещено от центра', en: 'Body off center', protocols: ['total_body'] },
  { code: 'body_asymmetry', ru: 'Асимметрия укладки', en: 'Positioning asymmetry', protocols: ['total_body'] },
  { code: 'total_body_incomplete_coverage', ru: 'Неполный охват тела', en: 'Incomplete total-body coverage', protocols: ['total_body'] },
  { code: 'edge_detection', ru: 'Ошибка определения контура', en: 'Edge detection error' },
  { code: 'soft_tissue_boundary', ru: 'Граница мягких тканей', en: 'Soft-tissue boundary' },
  { code: 'hyperdense_artifact', ru: 'Гиперденсный артефакт', en: 'Hyperdense artifact' },
  { code: 'hypodense_artifact', ru: 'Гиподенсный артефакт', en: 'Hypodense artifact' },
  { code: 'motion_artifact', ru: 'Двигательный артефакт', en: 'Motion artifact' },
  { code: 'external_object', ru: 'Внешний объект', en: 'External object' },
  { code: 'landmark_low_confidence', ru: 'Низкая уверенность ориентиров', en: 'Low landmark confidence' },
  { code: 'artifact_model_unavailable', ru: 'Модель артефактов недоступна', en: 'Artifact model unavailable' },
]

const defectsForProtocol = (protocol: DatasetProtocol) => defectCatalog.filter((item) => !item.protocols || item.protocols.includes(protocol))
const defectLabels = Object.fromEntries(defectCatalog.map(({ code, ru, en }) => [code, { ru, en }]))

const emptyLandmarks: AnnotationLandmark[] = ['L1', 'L2', 'L3', 'L4'].map((name) => ({ name, x: .5, y: .5, visible: false }))

const annotationProtocol = (protocol: DatasetProtocol): AnnotationDocument['protocol'] =>
  protocol === 'forearm_left' || protocol === 'forearm_right' ? 'unsupported' : protocol

const emptyAnnotation = (study: DatasetStudy): AnnotationDocument => ({
  schemaVersion: '1.0.0',
  studyId: study.studyId,
  patientGroupId: study.patientGroupId,
  deviceGroup: study.deviceGroup,
  protocol: annotationProtocol(study.protocol),
  evaluable: true,
  notEvaluableReason: '',
  overallAction: 'review',
  defects: defectsForProtocol(study.protocol).map(({ code }) => ({ code, present: false, severity: 'none' })),
  landmarks: emptyLandmarks,
  regions: [],
  expert: { readerId: '', readIndex: 1, confidence: 'medium', createdAt: new Date().toISOString(), adjudicated: false, comment: '' },
})

function MetricCard({ label, value, detail }: { label: string; value: string | number; detail: string }) {
  return <article className="dataset-metric"><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>
}

function ReadinessDashboard({ report, locale }: { report: ReadinessReport; locale: Locale }) {
  const percentage = (value: number) => `${Math.round(value * 100)}%`
  const gateText = (gate: ReadinessReport['gates'][number]) => locale === 'en' && readinessGateEnglish[gate.id]
    ? readinessGateEnglish[gate.id]
    : { label: gate.label, action: gate.nextAction }
  return (
    <section className="readiness-card" aria-labelledby="readiness-title">
      <div className="readiness-head">
        <div className="readiness-mark"><ShieldAlert size={22} /></div>
        <div><p className="eyebrow">Evidence & release gates</p><h2 id="readiness-title">{tx(locale, 'Готовность к клиническому пилоту', 'Clinical pilot readiness')}</h2><p>{tx(locale, 'Статус строится из проверяемых данных, разметки, моделей, безопасности и audit trail.', 'Status is derived from verifiable data, annotation, model, security, and audit evidence.')}</p></div>
        <span className={`readiness-stage is-${report.stage}`}>{report.stage === 'research' ? tx(locale, 'Только исследования', 'Research only') : tx(locale, 'Готов к silent pilot', 'Silent pilot ready')}</span>
      </div>
      <div className="readiness-summary">
        <article><span>{tx(locale, 'Критические блокеры', 'Critical blockers')}</span><strong>{report.criticalBlockerCount}</strong><small>{tx(locale, 'до silent pilot', 'before silent pilot')}</small></article>
        <article><span>Dataset</span><strong>{report.integrity.status === 'pass' ? tx(locale, 'Проверен', 'Valid') : tx(locale, 'Ошибка', 'Failed')}</strong><small>{report.datasetVersion}</small></article>
        <article><span>{tx(locale, 'Экспертное покрытие', 'Annotation coverage')}</span><strong>{percentage(report.agreement.annotationCoverage)}</strong><small>{tx(locale, `двойное чтение ${percentage(report.agreement.doubleReadCoverage)}`, `double read ${percentage(report.agreement.doubleReadCoverage)}`)}</small></article>
        <article><span>Audit trail</span><strong>{report.audit.eventCount === 0 ? tx(locale, 'Не начат', 'Not started') : report.audit.valid ? tx(locale, 'Проверен', 'Valid') : tx(locale, 'Нарушен', 'Invalid')}</strong><small>{report.audit.eventCount} {tx(locale, 'событий', 'events')}</small></article>
      </div>
      <details className="readiness-gates">
        <summary><UsersRound size={17} />{tx(locale, 'Показать все release gates', 'Show all release gates')}<span>{report.gates.filter((gate) => gate.status === 'pass').length}/{report.gates.length}</span></summary>
        <ul>{report.gates.map((gate) => { const copy = gateText(gate); return <li key={gate.id} className={`gate-${gate.status}`}><span className="gate-status">{gate.status === 'pass' ? <CheckCircle2 size={17} /> : <AlertTriangle size={17} />}</span><div><strong>{copy.label}</strong><small>{gate.evidence}</small><p>{gate.status === 'pass' ? tx(locale, 'Доказательство принято.', 'Evidence accepted.') : copy.action}</p></div><b>{gate.status === 'pass' ? tx(locale, 'ГОТОВО', 'PASS') : gate.status === 'warn' ? tx(locale, 'РИСК', 'WARN') : tx(locale, 'БЛОК', 'BLOCK')}</b></li> })}</ul>
      </details>
      <p className="readiness-intended-use"><strong>{tx(locale, 'Назначение:', 'Intended use:')}</strong> {report.intendedUse[locale]} <span>{report.policyVersion}</span></p>
    </section>
  )
}

function GeometryOverlay({ landmarks, regions }: { landmarks: AnnotationLandmark[]; regions: AnnotationRegion[] }) {
  return (
    <svg className="annotation-overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">
      {regions.map((region) => region.geometryType === 'box' && region.points.length === 2 ? (
        <rect key={region.name} x={Math.min(region.points[0][0], region.points[1][0]) * 1000} y={Math.min(region.points[0][1], region.points[1][1]) * 1000} width={Math.abs(region.points[1][0] - region.points[0][0]) * 1000} height={Math.abs(region.points[1][1] - region.points[0][1]) * 1000} rx="12" />
      ) : <polyline key={region.name} points={region.points.map(([x, y]) => `${x * 1000},${y * 1000}`).join(' ')} className={region.geometryType === 'polygon' ? 'is-polygon' : ''} />)}
      {landmarks.filter((item) => item.visible).map((item) => <g key={item.name}><circle cx={item.x * 1000} cy={item.y * 1000} r="13" /><text x={item.x * 1000 + 22} y={item.y * 1000 - 18}>{item.name}</text></g>)}
    </svg>
  )
}

function AnnotationEditor({ study, imageUrl, locale, onSaved }: { study: DatasetStudy; imageUrl?: string; locale: Locale; onSaved: (value: AnnotationDocument) => void }) {
  const latest = study.annotations?.[0]
  const [value, setValue] = useState<AnnotationDocument>(() => latest ?? emptyAnnotation(study))
  const [activeLandmark, setActiveLandmark] = useState('L1')
  const [regionTool, setRegionTool] = useState<'landmark' | 'box' | 'polyline'>('landmark')
  const [boxStart, setBoxStart] = useState<[number, number] | null>(null)
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [saveError, setSaveError] = useState('')
  const readerIdRef = useRef<HTMLInputElement>(null)
  const reasonRef = useRef<HTMLTextAreaElement>(null)
  const applicableDefects = defectsForProtocol(study.protocol)

  const updateLandmark = (name: string, patch: Partial<AnnotationLandmark>) => setValue((current) => ({
    ...current,
    landmarks: current.landmarks.map((item) => item.name === name ? { ...item, ...patch } : item),
  }))

  const placeGeometry = (event: React.MouseEvent<HTMLButtonElement>) => {
    if (event.detail === 0) return
    const bounds = event.currentTarget.getBoundingClientRect()
    const point: [number, number] = [
      Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
      Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
    ]
    if (regionTool === 'landmark') {
      updateLandmark(activeLandmark, { x: point[0], y: point[1], visible: true })
      const order = ['L1', 'L2', 'L3', 'L4']
      setActiveLandmark(order[(order.indexOf(activeLandmark) + 1) % order.length])
      return
    }
    if (regionTool === 'box') {
      if (!boxStart) { setBoxStart(point); return }
      const name = `${activeLandmark}_roi`
      setValue((current) => ({ ...current, regions: [...current.regions.filter((item) => item.name !== name), { name, geometryType: 'box', points: [boxStart, point] }] }))
      setBoxStart(null)
      return
    }
    const name = 'soft_tissue_boundary'
    setValue((current) => {
      const existing = current.regions.find((item) => item.name === name)
      const points = existing ? [...existing.points, point] : [[point[0], point[1]], [point[0], point[1]]] as [number, number][]
      return { ...current, regions: [...current.regions.filter((item) => item.name !== name), { name, geometryType: 'polyline', points }] }
    })
  }

  const setDefect = (code: string, patch: Partial<AnnotationDefect>) => setValue((current) => ({
    ...current,
    defects: current.defects.map((item) => item.code === code ? { ...item, ...patch } : item),
  }))

  const updateRegionPoint = (regionName: string, pointIndex: number, coordinateIndex: 0 | 1, rawValue: number) => {
    const bounded = Math.max(0, Math.min(1, Number.isFinite(rawValue) ? rawValue : 0))
    setValue((current) => ({
      ...current,
      regions: current.regions.map((region) => region.name === regionName ? {
        ...region,
        points: region.points.map((point, index) => index === pointIndex
          ? point.map((coordinate, axis) => axis === coordinateIndex ? bounded : coordinate) as [number, number]
          : point),
      } : region),
    }))
  }

  const save = async () => {
    if (!value.expert.readerId.trim()) { setSaveError(tx(locale, 'Укажите ID эксперта.', 'Enter a reader ID.')); setSaveState('error'); readerIdRef.current?.focus(); return }
    if (!value.evaluable && !value.notEvaluableReason.trim()) { setSaveError(tx(locale, 'Укажите причину, почему исследование нельзя оценить.', 'Explain why the study is not evaluable.')); setSaveState('error'); reasonRef.current?.focus(); return }
    setSaveState('saving'); setSaveError('')
    try {
      const saved = await saveDatasetAnnotation({ ...value, expert: { ...value.expert, createdAt: new Date().toISOString() } })
      setValue(saved); setSaveState('saved'); onSaved(saved)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : String(error)); setSaveState('error')
    }
  }

  const download = () => {
    const href = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a'); link.href = href; link.download = `${study.studyId}-annotation.json`; link.click(); URL.revokeObjectURL(href)
  }

  return (
    <section className="annotation-card" aria-labelledby="annotation-heading">
      <div className="workbench-section-head">
        <div><p className="eyebrow">Expert ground truth</p><h2 id="annotation-heading">{tx(locale, 'Экспертная разметка', 'Expert annotation')}</h2><p>{tx(locale, 'Классы дефектов и нормализованная анатомическая геометрия.', 'Defect classes and normalized anatomical geometry.')}</p></div>
        <div className="annotation-actions"><button className="secondary-button" onClick={download}><Download size={16} />JSON</button><button className="primary-button" onClick={save} disabled={saveState === 'saving'}>{saveState === 'saving' ? <LoaderCircle className="spinner" size={17} /> : <Save size={17} />}{tx(locale, 'Сохранить', 'Save')}</button></div>
      </div>

      <div className="annotation-grid">
        <div className="geometry-pane">
          <div className="geometry-toolbar" role="toolbar" aria-label={tx(locale, 'Инструменты разметки', 'Annotation tools')}>
            <button className={regionTool === 'landmark' ? 'is-active' : ''} aria-pressed={regionTool === 'landmark'} onClick={() => setRegionTool('landmark')}><Waypoints size={16} />{tx(locale, 'Точки', 'Points')}</button>
            <button className={regionTool === 'box' ? 'is-active' : ''} aria-pressed={regionTool === 'box'} onClick={() => setRegionTool('box')}><MousePointer2 size={16} />ROI</button>
            <button className={regionTool === 'polyline' ? 'is-active' : ''} aria-pressed={regionTool === 'polyline'} onClick={() => setRegionTool('polyline')}><SlidersHorizontal size={16} />{tx(locale, 'Контур', 'Contour')}</button>
            <label><span>{tx(locale, 'Позвонок', 'Vertebra')}</span><select value={activeLandmark} onChange={(event) => setActiveLandmark(event.target.value)}>{['L1', 'L2', 'L3', 'L4'].map((name) => <option key={name}>{name}</option>)}</select></label>
          </div>
          <div className="annotation-stage">
            {imageUrl ? <button type="button" className="annotation-image-target" onClick={placeGeometry} aria-describedby="geometry-help" aria-label={tx(locale, 'Поставить выбранную точку или угол области на изображении', 'Place the selected point or region corner on the image')}><img src={imageUrl} alt={tx(locale, 'Обработанное DXA для экспертной разметки', 'Processed DXA for expert annotation')} /><GeometryOverlay landmarks={value.landmarks} regions={value.regions} /></button> : <span>{tx(locale, 'Изображение недоступно', 'Image unavailable')}</span>}
          </div>
          <p id="geometry-help" className="geometry-help">{boxStart ? tx(locale, 'Выберите второй угол ROI.', 'Select the second ROI corner.') : tx(locale, 'Мышь — быстрое размещение. Точные координаты доступны ниже с клавиатуры.', 'Use the pointer for fast placement. Exact keyboard coordinates are available below.')}</p>
          <div className="coordinate-grid">
            {value.landmarks.map((item) => <fieldset key={item.name}><legend>{item.name}</legend><label><span>X</span><input type="number" min="0" max="1" step="0.01" value={item.x} onChange={(event) => updateLandmark(item.name, { x: Number(event.target.value), visible: true })} /></label><label><span>Y</span><input type="number" min="0" max="1" step="0.01" value={item.y} onChange={(event) => updateLandmark(item.name, { y: Number(event.target.value), visible: true })} /></label><label className="visibility-check"><input type="checkbox" checked={item.visible} onChange={(event) => updateLandmark(item.name, { visible: event.target.checked })} />{tx(locale, 'виден', 'visible')}</label></fieldset>)}
          </div>
          {value.regions.length > 0 && <div className="region-list"><h3>{tx(locale, 'Области и контуры', 'Regions and contours')}</h3>{value.regions.map((region) => <article key={region.name}><div className="region-heading"><div><strong>{region.name}</strong><span>{region.geometryType} · {region.points.length}</span></div><button className="icon-button" aria-label={tx(locale, `Удалить ${region.name}`, `Remove ${region.name}`)} onClick={() => setValue((current) => ({ ...current, regions: current.regions.filter((item) => item.name !== region.name) }))}><X size={16} /></button></div><div className="region-points">{region.points.map((point, index) => <fieldset key={`${region.name}-${index}`}><legend>P{index + 1}</legend><label><span>X</span><input aria-label={`${region.name} P${index + 1} X`} type="number" min="0" max="1" step="0.01" value={point[0]} onChange={(event) => updateRegionPoint(region.name, index, 0, Number(event.target.value))} /></label><label><span>Y</span><input aria-label={`${region.name} P${index + 1} Y`} type="number" min="0" max="1" step="0.01" value={point[1]} onChange={(event) => updateRegionPoint(region.name, index, 1, Number(event.target.value))} /></label></fieldset>)}</div></article>)}</div>}
        </div>

        <form className="expert-form" onSubmit={(event) => { event.preventDefault(); save() }}>
          <fieldset className="decision-fieldset"><legend>{tx(locale, 'Итоговое действие', 'Overall action')}</legend><div>{(['accept', 'review', 'repeat'] as AnnotationAction[]).map((action) => <label key={action} className={`decision-option decision-${action}`}><input type="radio" name="overall-action" value={action} checked={value.overallAction === action} onChange={() => setValue((current) => ({ ...current, overallAction: action }))} /><span>{action === 'accept' ? tx(locale, 'Принять', 'Accept') : action === 'review' ? tx(locale, 'Проверить', 'Review') : tx(locale, 'Повторить', 'Repeat')}</span></label>)}</div></fieldset>
          <label className="form-check"><input type="checkbox" checked={value.evaluable} onChange={(event) => setValue((current) => ({ ...current, evaluable: event.target.checked }))} /><span><strong>{tx(locale, 'Исследование можно оценить', 'Study is evaluable')}</strong><small>{tx(locale, 'Снимите отметку при неверном протоколе или критическом повреждении.', 'Clear for a wrong protocol or critically damaged study.')}</small></span></label>
          {!value.evaluable && <label className="form-field"><span>{tx(locale, 'Причина', 'Reason')}</span><textarea ref={reasonRef} aria-invalid={saveState === 'error' && !value.notEvaluableReason.trim()} aria-describedby="annotation-status" value={value.notEvaluableReason} onChange={(event) => setValue((current) => ({ ...current, notEvaluableReason: event.target.value }))} /></label>}
          <fieldset className="defects-fieldset"><legend>{tx(locale, 'Дефекты', 'Defects')}</legend><div className="defect-list">{applicableDefects.map((definition) => { const defect = value.defects.find((item) => item.code === definition.code); if (!defect) return null; return <article key={definition.code} className={defect.present ? 'is-present' : ''}><label><input type="checkbox" checked={defect.present} onChange={(event) => setDefect(defect.code, { present: event.target.checked, severity: event.target.checked ? 'minor' : 'none', action: event.target.checked ? 'review' : undefined })} /><span>{definition[locale]}</span></label><select aria-label={tx(locale, `Тяжесть: ${definition.ru}`, `Severity: ${definition.en}`)} value={defect.severity} disabled={!defect.present} onChange={(event) => setDefect(defect.code, { severity: event.target.value as AnnotationDefect['severity'] })}><option value="minor">{tx(locale, 'Незначительный', 'Minor')}</option><option value="major">{tx(locale, 'Значимый', 'Major')}</option><option value="critical">{tx(locale, 'Критический', 'Critical')}</option></select></article> })}</div></fieldset>
          <div className="expert-meta-grid"><label className="form-field"><span>{tx(locale, 'ID эксперта', 'Reader ID')}</span><input ref={readerIdRef} aria-invalid={saveState === 'error' && !value.expert.readerId.trim()} aria-describedby="annotation-status" value={value.expert.readerId} onChange={(event) => setValue((current) => ({ ...current, expert: { ...current.expert, readerId: event.target.value } }))} placeholder="reader-01" /></label><label className="form-field"><span>{tx(locale, 'Уверенность', 'Confidence')}</span><select value={value.expert.confidence} onChange={(event) => setValue((current) => ({ ...current, expert: { ...current.expert, confidence: event.target.value as AnnotationDocument['expert']['confidence'] } }))}><option value="low">{tx(locale, 'Низкая', 'Low')}</option><option value="medium">{tx(locale, 'Средняя', 'Medium')}</option><option value="high">{tx(locale, 'Высокая', 'High')}</option></select></label></div>
          <label className="form-field"><span>{tx(locale, 'Комментарий', 'Comment')}</span><textarea value={value.expert.comment} maxLength={2000} onChange={(event) => setValue((current) => ({ ...current, expert: { ...current.expert, comment: event.target.value } }))} placeholder={tx(locale, 'Наблюдения эксперта…', 'Reader observations…')} /></label>
          <label className="form-check"><input type="checkbox" checked={value.expert.adjudicated} onChange={(event) => setValue((current) => ({ ...current, expert: { ...current.expert, adjudicated: event.target.checked } }))} /><span><strong>{tx(locale, 'Согласованная разметка', 'Adjudicated annotation')}</strong><small>{tx(locale, 'Отмечать только после разрешения расхождений экспертов.', 'Use only after reader disagreements are resolved.')}</small></span></label>
          <div id="annotation-status" className="annotation-status" role="status" aria-live="polite">{saveState === 'saved' && <><CheckCircle2 size={17} />{tx(locale, 'Разметка сохранена', 'Annotation saved')}</>}{saveState === 'error' && <><AlertTriangle size={17} />{saveError}</>}</div>
        </form>
      </div>
    </section>
  )
}

export default function DatasetWorkbench({ locale }: { locale: Locale }) {
  const [summary, setSummary] = useState<DatasetSummary | null>(null)
  const [studies, setStudies] = useState<DatasetStudy[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [selected, setSelected] = useState<DatasetStudy | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<DatasetApiError | null>(null)
  const [selectedError, setSelectedError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [protocol, setProtocol] = useState('all')
  const [split, setSplit] = useState('all')
  const [viewKind, setViewKind] = useState<'processed' | 'raw'>('processed')
  const [assetIndex, setAssetIndex] = useState(0)
  const [rawChannel, setRawChannel] = useState(0)
  const [exclusionCount, setExclusionCount] = useState(0)
  const [readiness, setReadiness] = useState<ReadinessReport | null>(null)
  const [reloadKey, setReloadKey] = useState(0)
  const [selectedReloadKey, setSelectedReloadKey] = useState(0)

  useEffect(() => {
    let active = true
    Promise.all([getDatasetSummary(), getDatasetStudies(), getExclusionRegistry().catch(() => ({ exclusions: [], count: 0 })), getReadinessReport().catch(() => null)]).then(([summaryValue, list, exclusions, readinessValue]) => {
      if (!active) return
      const requestedStudyId = new URLSearchParams(window.location.search).get('study')
      const initialStudyId = list.studies.some((item) => item.studyId === requestedStudyId) ? requestedStudyId : list.studies[0]?.studyId
      setSummary(summaryValue); setStudies(list.studies); setSelectedId(initialStudyId ?? ''); setExclusionCount(exclusions.count); setReadiness(readinessValue); setError(null)
    }).catch((reason) => { if (active) setError(reason instanceof DatasetApiError ? reason : new DatasetApiError('DATASET_REQUEST_FAILED', String(reason), 0)) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [reloadKey])

  useEffect(() => {
    if (!selectedId) return
    let active = true
    getDatasetStudy(selectedId).then((value) => { if (active) { setSelected(value); setSelectedError(null); setAssetIndex(0); setRawChannel(0) } }).catch((reason) => { if (active) { setSelected(null); setSelectedError(reason instanceof Error ? reason.message : String(reason)) } })
    return () => { active = false }
  }, [selectedId, selectedReloadKey])

  const filtered = useMemo(() => studies.filter((study) => {
    const matchesQuery = !query.trim() || `${study.studyId} ${study.patientGroupId} ${study.protocolCode}`.toLowerCase().includes(query.toLowerCase())
    return matchesQuery && (protocol === 'all' || study.protocol === protocol) && (split === 'all' || study.split === split)
  }), [studies, query, protocol, split])

  const imageUrl = resolveDatasetUrl(selected ? viewKind === 'processed' ? selected.assets[assetIndex]?.url : selected.rawChannels[rawChannel]?.url : undefined)
  const datasetRootMissing = error?.code === 'DATASET_UNAVAILABLE'
  const connectionTitle = datasetRootMissing
    ? tx(locale, 'Обезличенный dataset не подключён', 'De-identified dataset is not connected')
    : tx(locale, 'Сервис данных недоступен', 'Dataset service is unavailable')
  const connectionDescription = datasetRootMissing
    ? tx(locale, 'Экспортируйте архив офлайн и укажите OSSEO_DATASET_ROOT при запуске backend.', 'Export the archive offline and set OSSEO_DATASET_ROOT when starting the backend.')
    : tx(locale, 'Запустите frontend и API единым контуром, затем повторите подключение.', 'Start the frontend and API together, then retry the connection.')
  const recoveryCommand = datasetRootMissing
    ? "OSSEO_PSEUDONYM_KEY='…' osseo-apex new.rar --output ./data/dataset"
    : 'make dev'
  const retryConnection = () => { setLoading(true); setError(null); setReloadKey((value) => value + 1) }
  const retrySelectedStudy = () => { setSelectedError(null); setSelectedReloadKey((value) => value + 1) }

  if (loading) return <section className="dataset-loading" role="status"><LoaderCircle className="spinner" size={30} /><h1>{tx(locale, 'Открываем Dataset Workbench', 'Opening Dataset Workbench')}</h1></section>
  if (error) return <section className="dataset-empty" aria-labelledby="dataset-empty-title"><span><Archive size={28} aria-hidden="true" /></span><p className="eyebrow">Dataset offline boundary</p><h1 id="dataset-empty-title">{connectionTitle}</h1><p>{connectionDescription}</p><code>{recoveryCommand}</code><button type="button" className="primary-button dataset-retry" onClick={retryConnection}><RefreshCw size={17} aria-hidden="true" />{tx(locale, 'Повторить подключение', 'Retry connection')}</button><details className="dataset-error-details"><summary>{tx(locale, 'Технические детали', 'Technical details')}</summary><code>{error.code}{error.status ? ` · HTTP ${error.status}` : ''}</code></details></section>

  return (
    <div className="dataset-workbench">
      <header className="dataset-hero">
        <div><p className="eyebrow">Dataset Workbench · v1</p><h1>{tx(locale, 'Контур данных и разметки', 'Data and annotation workspace')}</h1><p>{tx(locale, 'Обезличенные исследования, технический QC и экспертный ground truth — без физических утверждений о шести raw-каналах.', 'De-identified studies, technical QC, and expert ground truth—without physical claims about the six raw channels.')}</p></div>
        <div className="dataset-hero-actions"><span className="dataset-trust"><Check size={17} />{tx(locale, 'Прямые идентификаторы не экспортируются', 'Direct identifiers are not exported')}</span><a className="secondary-button" href={datasetExportUrl('coco')} download><Download size={16} />COCO</a><a className="secondary-button" href={datasetExportUrl('annotations')} download><Download size={16} />JSONL</a></div>
      </header>
      {summary && <section className="dataset-metrics" aria-label={tx(locale, 'Сводка датасета', 'Dataset summary')}><MetricCard label={tx(locale, 'Исследования', 'Studies')} value={summary.studyCount} detail={`${summary.patientGroupCount ?? '—'} ${tx(locale, 'групп пациентов', 'patient groups')}`} /><MetricCard label={tx(locale, 'Размечено', 'Annotated')} value={summary.annotatedStudyCount} detail={`${summary.annotationCount} ${tx(locale, 'экспертных чтений', 'expert reads')}`} /><MetricCard label={tx(locale, 'Требуют проверки', 'Needs review')} value={summary.reviewRequiredCount} detail={tx(locale, 'по техническому baseline', 'by technical baseline')} /><MetricCard label={tx(locale, 'Исключено', 'Excluded')} value={exclusionCount} detail="Secondary Capture · trainingEligible=false" /></section>}
      {readiness && <ReadinessDashboard report={readiness} locale={locale} />}
      <section className="dataset-browser" aria-labelledby="dataset-browser-heading">
        <aside className="dataset-list-pane">
          <div className="workbench-section-head"><div><p className="eyebrow">Manifest</p><h2 id="dataset-browser-heading">{tx(locale, 'Исследования', 'Studies')}</h2></div><strong>{filtered.length}</strong></div>
          <label className="search-field dataset-search"><Search size={17} /><span className="sr-only">{tx(locale, 'Поиск исследования', 'Search studies')}</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tx(locale, 'ID, группа или протокол', 'ID, group, or protocol')} /></label>
          <div className="dataset-filters"><label><span>{tx(locale, 'Протокол', 'Protocol')}</span><select value={protocol} onChange={(event) => setProtocol(event.target.value)}><option value="all">{tx(locale, 'Все', 'All')}</option>{Object.entries(protocolLabels).map(([key, label]) => <option key={key} value={key}>{label[locale]}</option>)}</select></label><label><span>Split</span><select value={split} onChange={(event) => setSplit(event.target.value)}><option value="all">{tx(locale, 'Все', 'All')}</option><option value="train">Train</option><option value="validation">Validation</option><option value="test">Test</option></select></label></div>
          <nav className="dataset-study-list" aria-label={tx(locale, 'Исследования датасета', 'Dataset studies')}>{filtered.map((study) => <button key={study.studyId} className={study.studyId === selectedId ? 'is-active' : ''} aria-current={study.studyId === selectedId ? 'true' : undefined} onClick={() => { setSelectedId(study.studyId); setSelectedError(null) }}><span className="dataset-study-status">{study.quality.reviewRequired ? <AlertTriangle size={15} /> : <CheckCircle2 size={15} />}</span><span><strong>{study.studyId}</strong><small>{protocolLabels[study.protocol][locale]} · {study.acquisitionYear ?? '—'}</small><i>{study.split ?? 'unassigned'} · {study.annotationCount} {tx(locale, 'разм.', 'reads')}</i></span></button>)}{filtered.length === 0 && <div className="dataset-filter-empty"><Search size={22} /><strong>{tx(locale, 'Ничего не найдено', 'No studies found')}</strong><p>{tx(locale, 'Измените запрос или сбросьте фильтры.', 'Change the query or clear the filters.')}</p><button type="button" className="secondary-button" onClick={() => { setQuery(''); setProtocol('all'); setSplit('all') }}>{tx(locale, 'Сбросить фильтры', 'Clear filters')}</button></div>}</nav>
        </aside>
        <div className="dataset-detail-pane">
          {studies.length === 0 ? <div className="dataset-no-selection"><Archive size={28} aria-hidden="true" /><p>{tx(locale, 'В manifest пока нет исследований', 'There are no studies in the manifest yet')}</p><small>{tx(locale, 'Добавьте обезличенный экспорт в OSSEO_DATASET_ROOT и повторите подключение.', 'Add a de-identified export to OSSEO_DATASET_ROOT and retry the connection.')}</small><button type="button" className="secondary-button" onClick={retryConnection}><RefreshCw size={16} aria-hidden="true" />{tx(locale, 'Обновить список', 'Refresh list')}</button></div> : selected && selected.studyId === selectedId ? <>
            <div className="dataset-study-head"><div><p className="eyebrow">{selected.protocolCode}</p><h2>{protocolLabels[selected.protocol][locale]}</h2><p>{selected.studyId} · {selected.patientGroupId} · APEX {selected.softwareVersion ?? '—'}</p></div><div className="dataset-badges"><span>{selected.split ?? 'unassigned'}</span><span className={selected.quality.reviewRequired ? 'needs-review' : ''}>{selected.technicalQc?.score ?? '—'}/100 QC</span></div></div>
            <div className="dataset-view-switch" role="group" aria-label={tx(locale, 'Представление сигнала', 'Signal view')}><button type="button" aria-pressed={viewKind === 'processed'} className={viewKind === 'processed' ? 'is-active' : ''} onClick={() => setViewKind('processed')}><ImageIcon size={16} />{tx(locale, 'Обработанные', 'Processed')}</button><button type="button" aria-pressed={viewKind === 'raw'} className={viewKind === 'raw' ? 'is-active' : ''} onClick={() => setViewKind('raw')}><Database size={16} />Raw × 6</button></div>
            {viewKind === 'processed' ? <div className="asset-tabs" role="group" aria-label={tx(locale, 'Обработанные изображения', 'Processed images')}>{selected.assets.map((asset, index) => <button type="button" key={asset.name} aria-pressed={assetIndex === index} className={assetIndex === index ? 'is-active' : ''} onClick={() => setAssetIndex(index)}>{asset.tag}</button>)}</div> : <div className="asset-tabs raw-tabs" role="group" aria-label={tx(locale, 'Каналы исходного сигнала', 'Raw signal channels')}>{selected.rawChannels.map((channel) => <button type="button" key={channel.index} aria-pressed={rawChannel === channel.index} className={rawChannel === channel.index ? 'is-active' : ''} onClick={() => setRawChannel(channel.index)}>{channel.index}</button>)}</div>}
            <div className="dataset-image-stage">{imageUrl ? <img src={imageUrl} alt={tx(locale, `${viewKind === 'processed' ? 'Обработанное изображение' : 'Raw-канал'} исследования ${selected.studyId}`, `${viewKind === 'processed' ? 'Processed image' : 'Raw channel'} for ${selected.studyId}`)} /> : <span>{tx(locale, 'Ассет недоступен', 'Asset unavailable')}</span>}<span className="dataset-image-label">{viewKind === 'raw' ? tx(locale, `Канал ${rawChannel} · физический смысл не подтверждён`, `Channel ${rawChannel} · physical meaning unverified`) : selected.assets[assetIndex]?.tag}</span></div>
            <div className="technical-qc-panel"><div><p className="eyebrow">Technical baseline</p><h3>{tx(locale, 'Автоматический pre-screening', 'Automated pre-screening')}</h3></div><strong>{selected.technicalQc?.score ?? '—'}<small>/100</small></strong><ul>{selected.technicalQc?.flags.length ? selected.technicalQc.flags.map((flag) => <li key={flag}><AlertTriangle size={14} />{flag}</li>) : <li><CheckCircle2 size={14} />{tx(locale, 'Технических флагов нет', 'No technical flags')}</li>}</ul><p>{tx(locale, 'Baseline не оценивает анатомию, BMD или корректность ROI.', 'The baseline does not assess anatomy, BMD, or ROI correctness.')}</p></div>
          </> : <div className="dataset-no-selection" role={selectedError ? 'alert' : 'status'}>{selectedError ? <AlertTriangle size={28} aria-hidden="true" /> : <LoaderCircle className="spinner" size={28} aria-hidden="true" />}<p>{selectedError ? tx(locale, 'Не удалось открыть исследование', 'Could not open the study') : tx(locale, 'Открываем исследование…', 'Opening study…')}</p>{selectedError && <button type="button" className="secondary-button" onClick={retrySelectedStudy}><RefreshCw size={16} aria-hidden="true" />{tx(locale, 'Повторить', 'Retry')}</button>}</div>}
        </div>
      </section>
      {selected && selected.studyId === selectedId && <AnnotationEditor key={selected.studyId} study={selected} imageUrl={resolveDatasetUrl(selected.assets[0]?.url)} locale={locale} onSaved={(annotation) => { setSelected((current) => current ? { ...current, annotations: [annotation, ...(current.annotations ?? []).filter((item) => !(item.expert.readerId === annotation.expert.readerId && item.expert.readIndex === annotation.expert.readIndex))], annotationCount: Math.max(1, current.annotationCount) } : current); setStudies((current) => current.map((item) => item.studyId === annotation.studyId ? { ...item, annotationCount: Math.max(1, item.annotationCount) } : item)) }} />}
      {selected && selected.studyId === selectedId && <DatasetLongitudinal key={`${selected.patientGroupId}-${selected.studyId}`} study={selected} studies={studies} locale={locale} defectLabels={defectLabels} />}
    </div>
  )
}
