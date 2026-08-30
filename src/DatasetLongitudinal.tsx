import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, CalendarRange, CheckCircle2, GitCompareArrows, History, LoaderCircle, Plus, Save } from 'lucide-react'
import type { DatasetStudy } from './services/dataset'
import {
  compareLongitudinal,
  getCrossCalibrations,
  getLscProfiles,
  getTimeline,
  saveLscProfile,
  saveMeasurement,
  type CrossCalibration,
  type LongitudinalComparison,
  type LongitudinalMeasurement,
  type LongitudinalProtocol,
  type LscProfile,
  type MeasurementSite,
  type ReviewStatus,
} from './services/longitudinal-api'
import type { Locale } from './types'

const tx = (locale: Locale, ru: string, en: string) => locale === 'ru' ? ru : en
const isSupported = (value: DatasetStudy['protocol']): value is LongitudinalProtocol => ['spine_pa', 'hip_left', 'hip_right'].includes(value)
const sitesFor = (protocol: LongitudinalProtocol): MeasurementSite[] => protocol === 'spine_pa' ? ['l1-l4'] : ['total-hip', 'femoral-neck']
const siteLabel: Record<MeasurementSite, string> = { 'l1-l4': 'L1–L4', 'total-hip': 'Total Hip', 'femoral-neck': 'Femoral Neck' }
const isoToday = () => new Date().toISOString().slice(0, 10)
const yearStart = () => `${new Date().getUTCFullYear()}-01-01`
const yearEnd = () => `${new Date().getUTCFullYear() + 3}-12-31`
const statusLabel = (value: ReviewStatus, locale: Locale) => ({ accept: tx(locale, 'Принято', 'Accepted'), review: tx(locale, 'Проверить', 'Review'), reject: tx(locale, 'Отклонено', 'Rejected') })[value]
const checkLabels: Record<string, [string, string]> = {
  patient: ['Один пациент', 'Same patient'], chronology: ['Хронология', 'Chronology'], protocol: ['Протокол и сторона', 'Protocol and side'],
  structured_bmd: ['Подтверждённые BMD', 'Confirmed BMD'], quality: ['Качество исследований', 'Study quality'],
  positioning: ['Сопоставимая укладка', 'Comparable positioning'], roi: ['Сопоставимые ROI', 'Comparable ROI'], device: ['Аппарат', 'Device'],
  cross_calibration: ['Cross-calibration', 'Cross-calibration'], lsc_profile: ['Профиль LSC', 'LSC profile'],
}

function checkDetail(check: LongitudinalComparison['checks'][number], locale: Locale) {
  if (locale === 'ru' || ['chronology', 'protocol', 'quality', 'positioning', 'roi'].includes(check.id)) return check.detail
  if (check.id === 'patient') return check.passed ? 'Patient group matches.' : 'The studies belong to different patient groups.'
  if (check.id === 'structured_bmd') return check.passed ? 'Both studies have confirmed BMD for a shared ROI.' : 'Confirmed BMD or a shared ROI is missing.'
  if (check.id === 'device') return check.detail.includes('изменился') ? 'Device changed; the cross-calibration gate determines eligibility.' : 'The device did not change.'
  if (check.id === 'cross_calibration') return check.passed ? 'Same device or a valid cross-calibration record.' : 'A valid cross-calibration record is missing.'
  if (check.id === 'lsc_profile') return check.passed ? 'The active facility LSC profile matches this study.' : 'The selected LSC profile does not match the date, device, operator, protocol, or ROI.'
  return check.detail
}

type MeasurementDraft = Omit<LongitudinalMeasurement, 'schemaVersion' | 'studyId' | 'patientGroupId' | 'protocol' | 'deviceGroup' | 'expertConfirmed' | 'sites'> & { bmd: Record<MeasurementSite, string> }
type ProfileDraft = Omit<LscProfile, 'sites'> & { sites: Array<{ site: MeasurementSite; percent: string }> }

function newMeasurementDraft(protocol: LongitudinalProtocol): MeasurementDraft {
  return {
    acquiredOn: isoToday(), facilityId: 'FACILITY-01', operatorGroup: 'OPERATORS-01',
    qualityStatus: 'accept', positioningStatus: 'accept', roiStatus: 'accept',
    source: 'manual-verified', sourceReference: '', confirmedBy: '',
    bmd: Object.fromEntries(sitesFor(protocol).map((site) => [site, ''])) as Record<MeasurementSite, string>,
  }
}

function newProfile(study: DatasetStudy): ProfileDraft {
  const protocol = study.protocol as LongitudinalProtocol
  return {
    schemaVersion: '2.0.0', profileId: `LSC-${study.deviceGroup.slice(3, 11)}-${protocol === 'spine_pa' ? 'SPINE' : 'HIP'}`,
    version: '1.0.0', facilityId: 'FACILITY-01', deviceGroup: study.deviceGroup,
    operatorGroup: 'OPERATORS-01', protocol, validFrom: yearStart(), validTo: yearEnd(),
    sites: sitesFor(protocol).map((site) => ({ site, percent: '' })),
    precisionStudyReference: '', approvedBy: '', status: 'active',
  }
}

function ComparisonResult({ value, locale }: { value: LongitudinalComparison; locale: Locale }) {
  const statusText = value.status === 'comparable' ? tx(locale, 'Сравнение допустимо', 'Comparison allowed') : value.status === 'review' ? tx(locale, 'Нужна проверка', 'Review required') : tx(locale, 'Сравнение заблокировано', 'Comparison blocked')
  return <section className={`real-trend-result is-${value.status}`} aria-labelledby="real-trend-result-title">
    <div className="real-trend-result-head"><span>{value.status === 'comparable' ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />}</span><div><p className="eyebrow">{value.engineVersion}</p><h3 id="real-trend-result-title">{statusText}</h3><small>{value.comparisonId} · {value.intervalMonths} {tx(locale, 'мес.', 'mo')}</small></div></div>
    {value.sites.length > 0 && <div className="real-trend-sites">{value.sites.map((site) => <article key={site.site}><span>{siteLabel[site.site]}</span><strong>{site.percentChange > 0 ? '+' : ''}{site.percentChange.toFixed(1)}%</strong><small>{site.baselineBmd.toFixed(3)} → {site.currentBmd.toFixed(3)} g/cm² · LSC {site.lscPercent.toFixed(1)}%</small><b>{site.status === 'not-comparable' ? tx(locale, 'Не интерпретируется', 'Not interpreted') : site.status === 'stable' ? tx(locale, 'Без значимых изменений', 'No significant change') : site.status === 'significant-gain' ? tx(locale, 'Значимый прирост', 'Significant gain') : tx(locale, 'Значимое снижение', 'Significant loss')}</b></article>)}</div>}
    <details className="real-trend-checks" open={value.status !== 'comparable'}><summary>{tx(locale, 'Проверки сопоставимости', 'Comparability checks')} <span>{value.checks.filter((item) => item.passed).length}/{value.checks.length}</span></summary><ul>{value.checks.map((check) => <li key={check.id} className={`is-${check.status}`}>{check.passed ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}<span><strong>{checkLabels[check.id]?.[locale === 'ru' ? 0 : 1] ?? check.label}</strong><small>{checkDetail(check, locale)}</small></span><b>{check.status === 'pass' ? tx(locale, 'Готово', 'Pass') : check.status === 'review' ? tx(locale, 'Проверить', 'Review') : tx(locale, 'Блок', 'Block')}</b></li>)}</ul></details>
  </section>
}

export default function DatasetLongitudinal({ study, studies, locale }: { study: DatasetStudy; studies: DatasetStudy[]; locale: Locale }) {
  const protocol = isSupported(study.protocol) ? study.protocol : null
  const [timeline, setTimeline] = useState<LongitudinalMeasurement[]>([])
  const [profiles, setProfiles] = useState<LscProfile[]>([])
  const [calibrations, setCalibrations] = useState<CrossCalibration[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState<'measurement' | 'profile' | 'compare' | ''>('')
  const [notice, setNotice] = useState('')
  const [measurementDraft, setMeasurementDraft] = useState<MeasurementDraft | null>(() => protocol ? newMeasurementDraft(protocol) : null)
  const [profileDraft, setProfileDraft] = useState<ProfileDraft | null>(() => protocol ? newProfile(study) : null)
  const [baselineId, setBaselineId] = useState('')
  const [profileId, setProfileId] = useState('')
  const [calibrationId, setCalibrationId] = useState('')
  const [result, setResult] = useState<LongitudinalComparison | null>(null)

  const refresh = async () => {
    const [timelineValue, profileValue, calibrationValue] = await Promise.all([getTimeline(study.patientGroupId), getLscProfiles(), getCrossCalibrations()])
    setTimeline(timelineValue.measurements); setProfiles(profileValue.profiles); setCalibrations(calibrationValue.calibrations)
  }

  useEffect(() => {
    let active = true
    Promise.all([getTimeline(study.patientGroupId), getLscProfiles(), getCrossCalibrations()]).then(([timelineValue, profileValue, calibrationValue]) => {
      if (!active) return
      setTimeline(timelineValue.measurements); setProfiles(profileValue.profiles); setCalibrations(calibrationValue.calibrations); setError('')
    }).catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : String(reason)) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [study.patientGroupId])

  const current = timeline.find((item) => item.studyId === study.studyId)
  const priorMeasurements = useMemo(() => current ? timeline.filter((item) => item.studyId !== current.studyId && item.protocol === current.protocol && item.acquiredOn < current.acquiredOn).reverse() : [], [current, timeline])
  const matchingProfiles = useMemo(() => current ? profiles.filter((item) => item.status === 'active' && item.protocol === current.protocol && item.deviceGroup === current.deviceGroup && item.facilityId === current.facilityId) : [], [current, profiles])
  const patientStudyCount = studies.filter((item) => item.patientGroupId === study.patientGroupId).length

  const saveMeasurementForm = async (event: React.FormEvent) => {
    event.preventDefault(); if (!measurementDraft || !protocol) return
    setBusy('measurement'); setNotice(''); setError('')
    try {
      const sites = sitesFor(protocol).map((site) => ({ site, bmd: Number(measurementDraft.bmd[site]) }))
      await saveMeasurement({ schemaVersion: '2.0.0', studyId: study.studyId, patientGroupId: study.patientGroupId, protocol, deviceGroup: study.deviceGroup, expertConfirmed: true, ...measurementDraft, sites })
      await refresh(); setNotice(tx(locale, 'Измерение сохранено и добавлено во временной ряд.', 'Measurement saved to the timeline.'))
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) } finally { setBusy('') }
  }

  const saveProfileForm = async (event: React.FormEvent) => {
    event.preventDefault(); if (!profileDraft) return
    setBusy('profile'); setNotice(''); setError('')
    try {
      const saved = await saveLscProfile({ ...profileDraft, sites: profileDraft.sites.map((item) => ({ ...item, percent: Number(item.percent) })) })
      await refresh(); setProfileId(saved.profileId); setNotice(tx(locale, 'Профиль LSC сохранён.', 'LSC profile saved.'))
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) } finally { setBusy('') }
  }

  const runComparison = async (event: React.FormEvent) => {
    event.preventDefault(); if (!baselineId || !profileId) { setError(tx(locale, 'Выберите baseline и профиль LSC.', 'Select a baseline and LSC profile.')); return }
    setBusy('compare'); setError(''); setNotice(''); setResult(null)
    try { setResult(await compareLongitudinal({ baselineStudyId: baselineId, currentStudyId: study.studyId, lscProfileId: profileId, crossCalibrationId: calibrationId || undefined })) } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) } finally { setBusy('') }
  }

  if (!protocol) return <section className="real-trend-card real-trend-unavailable" aria-labelledby="real-trend-heading"><History size={26} /><div><p className="eyebrow">Longitudinal registry</p><h2 id="real-trend-heading">{tx(locale, 'Динамика для этого протокола не поддерживается', 'This protocol is not supported for trends')}</h2><p>{tx(locale, 'Рабочий контур ограничен поясничным отделом и проксимальным отделом бедра.', 'The production workflow is limited to lumbar spine and proximal femur.')}</p></div></section>

  return <section className="real-trend-card" aria-labelledby="real-trend-heading">
    <div className="workbench-section-head"><div><p className="eyebrow">Structured BMD · LSC registry</p><h2 id="real-trend-heading">{tx(locale, 'Реальный анализ в динамике', 'Production longitudinal analysis')}</h2><p>{tx(locale, 'Расчёт выполняется только по подтверждённым BMD и после проверки пациента, дат, протокола, аппарата, качества, ROI и LSC.', 'The calculation uses confirmed BMD only, after patient, date, protocol, device, quality, ROI, and LSC checks.')}</p></div><span className="real-trend-safety"><AlertTriangle size={16} />{tx(locale, 'Не вычисляет BMD по изображению', 'Never derives BMD from pixels')}</span></div>
    <div className="real-trend-summary"><article><span>{tx(locale, 'Исследований пациента', 'Patient studies')}</span><strong>{patientStudyCount}</strong></article><article><span>{tx(locale, 'Измерений в registry', 'Registry measurements')}</span><strong>{timeline.length}</strong></article><article><span>{tx(locale, 'Профилей LSC', 'LSC profiles')}</span><strong>{profiles.length}</strong></article><article><span>{tx(locale, 'Текущее исследование', 'Current study')}</span><strong>{current ? tx(locale, 'Готово', 'Ready') : tx(locale, 'Нет BMD', 'No BMD')}</strong></article></div>
    {loading && <div className="real-trend-loading" role="status"><LoaderCircle className="spinner" size={22} />{tx(locale, 'Загружаем registry…', 'Loading registry…')}</div>}
    <div className="real-trend-status" role="status" aria-live="polite">{error && <span className="is-error"><AlertTriangle size={16} />{error}</span>}{notice && <span className="is-success"><CheckCircle2 size={16} />{notice}</span>}</div>

    {!loading && <div className="real-trend-grid">
      <div className="real-trend-setup">
        <details className="real-trend-form-card" open={!current}>
          <summary><span><Plus size={18} />{current ? tx(locale, 'Обновить структурированное измерение', 'Update structured measurement') : tx(locale, 'Добавить структурированное измерение', 'Add structured measurement')}</span><small>{study.studyId}</small></summary>
          {measurementDraft && <form onSubmit={saveMeasurementForm}>
            <div className="real-trend-fields"><label><span>{tx(locale, 'Дата исследования', 'Study date')}</span><input type="date" required value={measurementDraft.acquiredOn} onChange={(event) => setMeasurementDraft({ ...measurementDraft, acquiredOn: event.target.value })} /></label><label><span>{tx(locale, 'Учреждение', 'Facility')}</span><input required pattern="[A-Z][A-Z0-9._-]{2,79}" value={measurementDraft.facilityId} onChange={(event) => setMeasurementDraft({ ...measurementDraft, facilityId: event.target.value.toUpperCase() })} /></label><label><span>{tx(locale, 'Группа операторов', 'Operator group')}</span><input pattern="[A-Z][A-Z0-9._-]{2,79}" value={measurementDraft.operatorGroup ?? ''} onChange={(event) => setMeasurementDraft({ ...measurementDraft, operatorGroup: event.target.value.toUpperCase() })} /></label><label><span>{tx(locale, 'Источник BMD', 'BMD source')}</span><select value={measurementDraft.source} onChange={(event) => setMeasurementDraft({ ...measurementDraft, source: event.target.value as LongitudinalMeasurement['source'] })}><option value="manual-verified">{tx(locale, 'Ручной ввод по отчёту', 'Manual from report')}</option><option value="vendor-structured">Vendor structured</option><option value="dicom-sr">DICOM SR</option></select></label>
              {sitesFor(protocol).map((site) => <label key={site}><span>BMD {siteLabel[site]}, g/cm²</span><input type="number" inputMode="decimal" min="0.001" max="5" step="0.001" required value={measurementDraft.bmd[site] ?? ''} onChange={(event) => setMeasurementDraft({ ...measurementDraft, bmd: { ...measurementDraft.bmd, [site]: event.target.value } })} /></label>)}
              <label><span>{tx(locale, 'Качество', 'Quality')}</span><select value={measurementDraft.qualityStatus} onChange={(event) => setMeasurementDraft({ ...measurementDraft, qualityStatus: event.target.value as ReviewStatus })}>{(['accept', 'review', 'reject'] as ReviewStatus[]).map((status) => <option key={status} value={status}>{statusLabel(status, locale)}</option>)}</select></label><label><span>{tx(locale, 'Укладка', 'Positioning')}</span><select value={measurementDraft.positioningStatus} onChange={(event) => setMeasurementDraft({ ...measurementDraft, positioningStatus: event.target.value as ReviewStatus })}>{(['accept', 'review', 'reject'] as ReviewStatus[]).map((status) => <option key={status} value={status}>{statusLabel(status, locale)}</option>)}</select></label><label><span>ROI</span><select value={measurementDraft.roiStatus} onChange={(event) => setMeasurementDraft({ ...measurementDraft, roiStatus: event.target.value as ReviewStatus })}>{(['accept', 'review', 'reject'] as ReviewStatus[]).map((status) => <option key={status} value={status}>{statusLabel(status, locale)}</option>)}</select></label><label><span>{tx(locale, 'Ссылка на источник', 'Source reference')}</span><input required minLength={3} value={measurementDraft.sourceReference} onChange={(event) => setMeasurementDraft({ ...measurementDraft, sourceReference: event.target.value })} placeholder="report-sha256:…" /></label><label><span>{tx(locale, 'ID подтвердившего эксперта', 'Confirming reader ID')}</span><input required minLength={2} value={measurementDraft.confirmedBy} onChange={(event) => setMeasurementDraft({ ...measurementDraft, confirmedBy: event.target.value })} placeholder="reader-01" /></label></div>
            <p className="real-trend-form-note">{tx(locale, 'Не указывайте ФИО пациента или номер истории болезни. Эксперт будет сохранён как HMAC-псевдоним.', 'Do not enter patient names or medical record numbers. The reader is stored as an HMAC pseudonym.')}</p><button className="primary-button" disabled={busy === 'measurement'}>{busy === 'measurement' ? <LoaderCircle className="spinner" size={17} /> : <Save size={17} />}{tx(locale, 'Сохранить измерение', 'Save measurement')}</button>
          </form>}
        </details>

        <details className="real-trend-form-card" open={profiles.length === 0}>
          <summary><span><Plus size={18} />{tx(locale, 'Добавить профиль точности LSC', 'Add LSC precision profile')}</span><small>{profileDraft?.profileId}</small></summary>
          {profileDraft && <form onSubmit={saveProfileForm}><div className="real-trend-fields"><label><span>ID профиля</span><input required pattern="LSC-[A-Z0-9._-]{4,76}" value={profileDraft.profileId} onChange={(event) => setProfileDraft({ ...profileDraft, profileId: event.target.value.toUpperCase() })} /></label><label><span>{tx(locale, 'Версия', 'Version')}</span><input required pattern="[0-9]+\.[0-9]+\.[0-9]+" value={profileDraft.version} onChange={(event) => setProfileDraft({ ...profileDraft, version: event.target.value })} /></label><label><span>{tx(locale, 'Учреждение', 'Facility')}</span><input required pattern="[A-Z][A-Z0-9._-]{2,79}" value={profileDraft.facilityId} onChange={(event) => setProfileDraft({ ...profileDraft, facilityId: event.target.value.toUpperCase() })} /></label><label><span>{tx(locale, 'Группа операторов', 'Operator group')}</span><input pattern="[A-Z][A-Z0-9._-]{2,79}" value={profileDraft.operatorGroup ?? ''} onChange={(event) => setProfileDraft({ ...profileDraft, operatorGroup: event.target.value.toUpperCase() })} /></label><label><span>{tx(locale, 'Действует с', 'Valid from')}</span><input type="date" required value={profileDraft.validFrom} onChange={(event) => setProfileDraft({ ...profileDraft, validFrom: event.target.value })} /></label><label><span>{tx(locale, 'Действует до', 'Valid to')}</span><input type="date" required value={profileDraft.validTo} onChange={(event) => setProfileDraft({ ...profileDraft, validTo: event.target.value })} /></label>{profileDraft.sites.map((site, index) => <label key={site.site}><span>LSC {siteLabel[site.site]}, %</span><input type="number" inputMode="decimal" min="0.1" max="25" step="0.1" required value={site.percent} placeholder={site.site === 'l1-l4' ? '5.3' : site.site === 'total-hip' ? '5.0' : '6.9'} onChange={(event) => setProfileDraft({ ...profileDraft, sites: profileDraft.sites.map((item, itemIndex) => itemIndex === index ? { ...item, percent: event.target.value } : item) })} /></label>)}<label><span>{tx(locale, 'Отчёт precision study', 'Precision study reference')}</span><input required minLength={3} value={profileDraft.precisionStudyReference} onChange={(event) => setProfileDraft({ ...profileDraft, precisionStudyReference: event.target.value })} placeholder="precision-study-2026-01" /></label><label><span>{tx(locale, 'Утвердил', 'Approved by')}</span><input required minLength={2} value={profileDraft.approvedBy} onChange={(event) => setProfileDraft({ ...profileDraft, approvedBy: event.target.value })} placeholder="physicist-01" /></label></div><p className="real-trend-form-note">{tx(locale, 'Ориентиры ISCD 5,3/5,0/6,9% показаны только как placeholder и не подставляются. Введите результаты precision assessment учреждения.', 'ISCD 5.3/5.0/6.9% references are placeholders only and are never submitted by default. Enter the facility precision assessment results.')}</p><button className="secondary-button" disabled={busy === 'profile'}>{busy === 'profile' ? <LoaderCircle className="spinner" size={17} /> : <Save size={17} />}{tx(locale, 'Сохранить профиль LSC', 'Save LSC profile')}</button></form>}
        </details>
      </div>

      <div className="real-trend-comparison">
        <div className="real-trend-timeline"><div><CalendarRange size={19} /><h3>{tx(locale, 'Временной ряд пациента', 'Patient timeline')}</h3></div>{timeline.length ? <ol>{timeline.map((item) => <li key={item.studyId} className={item.studyId === study.studyId ? 'is-current' : ''}><span>{item.acquiredOn}</span><strong>{item.studyId}</strong><small>{item.sites.map((site) => `${siteLabel[site.site]} ${site.bmd.toFixed(3)}`).join(' · ')} g/cm²</small></li>)}</ol> : <p>{tx(locale, 'Добавьте первое подтверждённое измерение.', 'Add the first confirmed measurement.')}</p>}</div>
        <form className="real-trend-compare-form" onSubmit={runComparison}><div><GitCompareArrows size={20} /><h3>{tx(locale, 'Сравнить с baseline', 'Compare with baseline')}</h3></div><label><span>Baseline</span><select required disabled={!current} value={baselineId} onChange={(event) => setBaselineId(event.target.value)}><option value="">{tx(locale, 'Выберите исследование', 'Select a study')}</option>{priorMeasurements.map((item) => <option key={item.studyId} value={item.studyId}>{item.acquiredOn} · {item.studyId}</option>)}</select></label><label><span>{tx(locale, 'Профиль LSC учреждения', 'Facility LSC profile')}</span><select required disabled={!current} value={profileId} onChange={(event) => setProfileId(event.target.value)}><option value="">{tx(locale, 'Выберите профиль', 'Select a profile')}</option>{matchingProfiles.map((item) => <option key={item.profileId} value={item.profileId}>{item.profileId} · v{item.version}</option>)}</select></label>{current && baselineId && timeline.find((item) => item.studyId === baselineId)?.deviceGroup !== current.deviceGroup && <label><span>Cross-calibration</span><select value={calibrationId} onChange={(event) => setCalibrationId(event.target.value)}><option value="">{tx(locale, 'Нет записи — сравнение будет заблокировано', 'No record — comparison will be blocked')}</option>{calibrations.map((item) => <option key={item.calibrationId} value={item.calibrationId}>{item.calibrationId}</option>)}</select></label>}<button className="primary-button" disabled={!current || busy === 'compare'}>{busy === 'compare' ? <LoaderCircle className="spinner" size={17} /> : <GitCompareArrows size={17} />}{tx(locale, 'Проверить и рассчитать', 'Validate and calculate')}</button>{current && priorMeasurements.length === 0 && <p>{tx(locale, 'Нет более раннего измерения того же протокола. Добавьте BMD для предыдущего исследования пациента.', 'No earlier measurement has the same protocol. Add BMD for a prior patient study.')}</p>}</form>
        {result && <ComparisonResult value={result} locale={locale} />}
      </div>
    </div>}
  </section>
}
