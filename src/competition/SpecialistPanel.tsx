export type SpecialistAssessment = {
  mode: 'shadow'
  affects_decision: false
  status?: 'unavailable'
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

export default function SpecialistPanel({ result }: { result?: SpecialistAssessment }) {
  if (!result) return null
  return <section aria-label="Экспериментальная оценка качества">
    <h3>Экспериментальная оценка качества</h3>
    <p>Дополнительная модель проходит проверку. Её результат не меняет основное заключение и экспертные правки.</p>
    {result.status === 'unavailable' || !result.predictions
      ? <p>Дополнительную оценку получить не удалось. Используйте основной результат и проверьте снимок вручную.</p>
      : <details><summary>Посмотреть результат дополнительной модели</summary>
        <ul>{Object.entries(result.predictions).filter(([, p]) => p.status !== 'not_applicable').map(([name, p]) =>
          <li key={name}><strong>{labels[name] || name}:</strong> {statuses[p.status] || 'оценка недоступна'}</li>)}</ul>
      </details>}
  </section>
}
