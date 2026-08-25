import type { Study } from '../types'

export const isDicomFile = (file: File) => /\.(dcm|dicom)$/i.test(file.name)
export const MAX_DICOM_BYTES = 100 * 1024 * 1024

export async function analyzeDicom(file: File): Promise<Study> {
  await new Promise((resolve) => window.setTimeout(resolve, 1450))
  const now = new Date()
  const suffix = String(Math.floor(1000 + Math.random() * 8999))

  return {
    id: `ST-${suffix}`,
    patientId: `P-${Math.floor(10000 + Math.random() * 89999)}`,
    filename: file.name,
    acquiredAt: now.toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' }).replace(',', ' ·'),
    type: 'spine',
    status: 'passed',
    score: 94,
    confidence: 97,
    device: 'DICOM device',
    institution: { ru: 'Не указано', en: 'Not provided' },
    operator: '—',
    accessionNumber: `LOCAL-${suffix}`,
    seriesUid: `2.25.${Date.now()}${suffix}`,
    criteria: [
      { id: 'position', title: { ru: 'Позиционирование', en: 'Positioning' }, detail: { ru: 'Положение анатомической области соответствует протоколу.', en: 'Anatomical region positioning matches the protocol.' }, status: 'pass', confidence: 98 },
      { id: 'coverage', title: { ru: 'Анатомический охват', en: 'Anatomical coverage' }, detail: { ru: 'Целевые структуры полностью входят в поле сканирования.', en: 'Target structures are fully included in the scan area.' }, status: 'pass', confidence: 97 },
      { id: 'markup', title: { ru: 'Анатомическая разметка', en: 'Anatomical markup' }, detail: { ru: 'Границы областей интереса определены корректно.', en: 'Region-of-interest boundaries are correctly defined.' }, status: 'pass', confidence: 96 },
      { id: 'artifacts', title: { ru: 'Артефакты', en: 'Artifacts' }, detail: { ru: 'Значимых артефактов не обнаружено.', en: 'No significant artifacts detected.' }, status: 'pass', confidence: 97 },
    ],
    recommendation: {
      ru: 'Исследование прошло демонстрационную автоматическую проверку. Перед клиническим использованием требуется валидация продукционной модели.',
      en: 'The study passed the demo automated review. Production-model validation is required before clinical use.',
    },
  }
}
