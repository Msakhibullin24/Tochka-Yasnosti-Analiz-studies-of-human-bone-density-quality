import { useState, type FormEvent } from 'react'
import { buildDynamicsDescription, type DynamicsResponse } from './dynamicsDescription'

export default function CompetitionDynamics() {
  const [baseline, setBaseline] = useState<File | null>(null)
  const [followup, setFollowup] = useState<File | null>(null)
  const [identityChecked, setIdentityChecked] = useState(false)
  const [baselineBmd, setBaselineBmd] = useState('')
  const [followupBmd, setFollowupBmd] = useState('')
  const [lsc, setLsc] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<DynamicsResponse | null>(null)

  const compare = async (event: FormEvent) => {
    event.preventDefault()
    setError(''); setResult(null)
    if (!baseline || !followup || !identityChecked) {
      setError('Выберите два DICOM одного пациента по порядку визитов и подтвердите сопоставление документов.')
      return
    }
    const anyBmd = Boolean(baselineBmd || followupBmd || lsc)
    if (anyBmd && (!baselineBmd || !followupBmd || Number(baselineBmd) <= 0 || Number(followupBmd) <= 0 ||
      (lsc !== '' && Number(lsc) <= 0))) {
      setError('Для расчёта МПК укажите оба положительных значения; LSC должен быть положительным.')
      return
    }
    const payload = new FormData()
    payload.append('baseline', baseline)
    payload.append('followup', followup)
    if (baselineBmd && followupBmd) {
      payload.append('baseline_bmd', baselineBmd)
      payload.append('followup_bmd', followupBmd)
      if (lsc) payload.append('lsc_percent', lsc)
    }
    setBusy(true)
    try {
      const response = await fetch('/api/v1/compare', { method: 'POST', body: payload })
      const body = await response.json()
      if (!response.ok || body.processing_status !== 'Success') throw new Error(
        typeof body.error_message === 'string' ? body.error_message :
          typeof body.detail === 'string' ? body.detail : 'Сравнение не выполнено. Проверьте DICOM и повторите.')
      setResult(body as DynamicsResponse)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Не удалось сравнить исследования.')
    } finally { setBusy(false) }
  }

  const description = result ? buildDynamicsDescription(result) : ''
  const download = () => {
    if (!description) return
    const url = URL.createObjectURL(new Blob([description], { type: 'text/plain;charset=utf-8' }))
    const link = document.createElement('a')
    link.href = url; link.download = 'dxa-dynamics-description.txt'
    document.body.appendChild(link); link.click(); link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 0)
  }

  return <details className="qc-dynamics">
    <summary>Анализ в динамике двух DXA</summary>
    <p>Выберите исходный и повторный DICOM. МПК вводится из подтверждённых документов; по пикселям она не вычисляется.</p>
    <form onSubmit={compare}>
      <label>Исходное исследование<input type="file" accept=".dcm,.dicom,application/dicom" required onChange={(event) => { setBaseline(event.target.files?.[0] ?? null); setResult(null) }} /></label>
      <label>Повторное исследование<input type="file" accept=".dcm,.dicom,application/dicom" required onChange={(event) => { setFollowup(event.target.files?.[0] ?? null); setResult(null) }} /></label>
      <label><input type="checkbox" checked={identityChecked} onChange={(event) => { setIdentityChecked(event.target.checked); setResult(null) }} /> По документам проверено: один пациент, повторное исследование выполнено позже.</label>
      <div className="qc-dynamics-values">
        <label>МПК исходная, g/cm²<input type="number" min="0.001" step="0.001" inputMode="decimal" value={baselineBmd} onChange={(event) => { setBaselineBmd(event.target.value); setResult(null) }} /></label>
        <label>МПК повторная, g/cm²<input type="number" min="0.001" step="0.001" inputMode="decimal" value={followupBmd} onChange={(event) => { setFollowupBmd(event.target.value); setResult(null) }} /></label>
        <label>LSC учреждения, %<input type="number" min="0.1" step="0.1" inputMode="decimal" value={lsc} onChange={(event) => { setLsc(event.target.value); setResult(null) }} /></label>
      </div>
      <button type="submit" className="primary-button" disabled={busy}>{busy ? 'Сравниваем…' : 'Сравнить исследования'}</button>
    </form>
    {error && <p className="qc-error" role="alert">{error}</p>}
    {result && <section aria-label="Результат сравнения" className="qc-dynamics-result">
      <h3>{result.verdict_ru}</h3><pre>{description}</pre>
      <button type="button" onClick={download}>Скачать описание динамики</button>
    </section>}
    <p>Пороги укладки пока не валидированы на парных DXA. Сравнение не подтверждает диагноз или эффективность лечения.</p>
  </details>
}
