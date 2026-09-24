export type SpecialistAssessment = {
  mode: 'shadow'
  affects_decision: false
  model_id?: string
  backbone?: string
  responsibility?: string
  region_scope?: string
  status?: 'ok' | 'unavailable' | 'not_applicable'
  predictions?: Record<string, { score: number | null; threshold: number | null; status: string }>
}

const labels: Record<string, string> = {
  quality: 'Общая оценка качества', spine_coverage: 'Охват позвоночника',
  spine_axis: 'Ось позвоночника', spine_artifact: 'Посторонние предметы',
  hip_position_rotation: 'Укладка бедра', hip_roi_coverage: 'Область интереса бедра',
}
const statuses: Record<string, string> = {
  pass: 'нарушение не обнаружено', fail: 'предполагается нарушение',
  undetermined: 'недостаточно данных для оценки',
}

const modelLabels: Record<string, string> = {
  spine_convnext: 'ConvNeXt: позвоночник', hip_convnext: 'ConvNeXt: бедро',
  general_efficientnet: 'EfficientNet: общая оценка',
}

function ModelResult({ name, result }: { name: string; result: SpecialistAssessment }) {
  return <details><summary>{modelLabels[name] || name}</summary>
    {result.status === 'not_applicable'
      ? <p>Модель предназначена для другой анатомической области.</p>
      : result.status === 'unavailable' || !result.predictions
        ? <p>Оценка этой модели недоступна.</p>
        : <ul>{Object.entries(result.predictions).filter(([, p]) => p.status !== 'not_applicable').map(([key, p]) =>
          <li key={key}><strong>{labels[key] || key}:</strong> {statuses[p.status] || 'оценка недоступна'}
            {p.score !== null ? ` · балл ${p.score.toFixed(3)}` : ''}
            {p.threshold !== null ? ` · порог ${p.threshold.toFixed(3)}` : ''}</li>)}</ul>}
  </details>
}

export default function SpecialistPanel({ result, results }: { result?: SpecialistAssessment; results?: Record<string, SpecialistAssessment> }) {
  if (!result && !results) return null
  return <section aria-label="Экспериментальная оценка качества">
    <h3>Независимые оценки моделей</h3>
    <p>Каждая модель использует собственные веса и выдаёт свой результат. Эти оценки не меняют основной конкурсный вывод.</p>
    {result && <ModelResult name="legacy_specialist" result={result} />}
    {results && Object.entries(results).map(([name, prediction]) =>
      <ModelResult key={name} name={name} result={prediction} />)}
  </section>
}
