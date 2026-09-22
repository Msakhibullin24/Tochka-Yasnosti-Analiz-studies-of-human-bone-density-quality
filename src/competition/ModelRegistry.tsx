import { useEffect, useRef, useState } from 'react'

type Capability = { id: string; readiness: string; artifacts_verified: boolean }
type Inventory = { configured: boolean; capabilities: Capability[] }
const descriptions: Record<string, { name: string; purpose: string }> = {
  dxa_to_3d: { name: 'DXA-to-3D', purpose: 'Исследовательский выход формы позвоночника. Нумерация позвонков и геометрия ещё не проверены.' },
  convnextv2_tiny: { name: 'ConvNeXt-V2', purpose: 'Основа обученных QC-кандидатов. Активная дополнительная оценка показана в карточке снимка.' },
  efficientnet_b4: { name: 'EfficientNet-B4', purpose: 'Обучен экспериментальный QC-кандидат для сравнения метрик.' },
  yolo26x: { name: 'YOLO26x Detection / Segmentation', purpose: 'Общие веса COCO. Для имплантов и артефактов нужна разметка DXA и дообучение.' },
  yolo26x_pose: { name: 'YOLO26x Pose', purpose: 'Начальные веса для обучения ориентиров. Готовая модель костных ориентиров пока не обучена.' },
  totalbody_105: { name: '105 ориентиров всего тела', purpose: 'Отдельный исследовательский анализ whole-body air-ratio. Не применяется к локальным снимкам бедра и поясницы.' },
  nhanes: { name: 'NHANES', purpose: 'Таблицы показателей и кодов качества. Обучающих изображений в этой поставке нет.' },
}

export default function ModelRegistry() {
  const [data, setData] = useState<Inventory | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  async function refresh() {
    controller.current?.abort()
    const request = new AbortController()
    controller.current = request
    setBusy(true); setMessage('Проверка файлов моделей…')
    try {
      const response = await fetch('/api/v1/specialists', { signal: request.signal })
      if (!response.ok) throw new Error('Inventory unavailable')
      const result = await response.json() as Inventory
      if (!Array.isArray(result.capabilities)) throw new Error('Invalid inventory')
      if (!request.signal.aborted) {
        setData(result)
        setMessage(result.configured ? 'Проверка завершена. Наличие файлов не подтверждает качество модели.' : 'Каталог моделей не настроен на этом сервере.')
      }
    } catch {
      if (!request.signal.aborted) { setData(null); setMessage('Не удалось проверить модели. Повторите проверку после восстановления подключения.') }
    } finally {
      if (!request.signal.aborted) setBusy(false)
    }
  }
  return <details className="qc-batches">
    <summary>Модели и данные проекта</summary>
    <div className="qc-history">
      <p>Здесь указаны компоненты проекта и назначение скачанных моделей. Исследовательские модели не меняют основное заключение.</p>
      <button className="qc-text-button" disabled={busy} onClick={refresh}>Проверить файлы моделей</button>
      <p role="status">{message}</p>
      {data?.capabilities.map(item => <article key={item.id}>
        <h3>{descriptions[item.id]?.name || item.id}</h3>
        <p>{item.artifacts_verified ? 'Файлы из каталога проверены.' : 'Часть файлов отсутствует или повреждена.'}</p>
        <p>{descriptions[item.id]?.purpose || 'Назначение модели уточняется.'}</p>
      </article>)}
    </div>
  </details>
}
