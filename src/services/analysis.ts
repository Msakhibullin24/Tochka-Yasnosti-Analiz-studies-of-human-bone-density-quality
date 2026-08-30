import type { Criterion, CriterionStatus, LocalizedText, Study, StudyStatus, StudyType } from '../types'
import { parseDicomBytes, SUPPORTED_UNCOMPRESSED_TRANSFER_SYNTAXES, type ParsedDicom } from './dicom'

export const MAX_DICOM_BYTES = 100 * 1024 * 1024
export const ANALYSIS_VERSION = 'technical-screening-0.3.0'
export const CRITERIA_VERSION = 'DXA-QC-2026.1'

export type DicomErrorCode = 'empty-file' | 'file-too-large' | 'invalid-dicom'

export class DicomAnalysisError extends Error {
  constructor(public readonly code: DicomErrorCode, message: string) {
    super(message)
    this.name = 'DicomAnalysisError'
  }
}

export const isDicomFile = (file: File) =>
  /\.(dcm|dicom)$/i.test(file.name) || ['application/dicom', 'application/dicom+json'].includes(file.type)

const t = (ru: string, en: string): LocalizedText => ({ ru, en })

const fallbackHash = (input: string) => {
  let hash = 0x811c9dc5
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16).padStart(8, '0').toUpperCase()
}

const stableHash = async (input: string) => {
  if (!globalThis.crypto?.subtle) return fallbackHash(input)
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(input))
  return Array.from(new Uint8Array(digest).slice(0, 4), (byte) => byte.toString(16).padStart(2, '0')).join('').toUpperCase()
}

const inferStudyType = (dicom: ParsedDicom, filename = ''): { type: StudyType; recognized: boolean } => {
  const haystack = `${dicom.bodyPart} ${dicom.description} ${dicom.protocolName} ${filename}`.toUpperCase()
  if (/HIP|FEMUR|FEMOR|PROXIMAL|БЕДР|ТАЗОБЕДР/.test(haystack)) return { type: 'hip', recognized: true }
  if (/SPINE|LUMBAR|LSP|L[1-5]|ПОЗВ|ПОЯСНИЧ/.test(haystack)) return { type: 'spine', recognized: true }
  return { type: 'spine', recognized: false }
}

const formatDicomDate = (date: string, time: string) => {
  const dateLabel = /^\d{8}$/.test(date) ? `${date.slice(6, 8)}.${date.slice(4, 6)}.${date.slice(0, 4)}` : new Date().toLocaleDateString('ru-RU')
  const timeLabel = /^\d{4}/.test(time) ? `${time.slice(0, 2)}:${time.slice(2, 4)}` : new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
  return `${dateLabel} · ${timeLabel}`
}

const criterion = (id: string, title: LocalizedText, detail: LocalizedText, status: CriterionStatus, confidence: number): Criterion =>
  ({ id, title, detail, status, confidence })

function buildCriteria(dicom: ParsedDicom, protocolRecognized: boolean): Criterion[] {
  const hasGeometry = Boolean(dicom.rows && dicom.columns)
  const integrityStatus: CriterionStatus = dicom.pixelDataPresent && hasGeometry ? 'pass' : 'fail'
  const supportedPixels = Boolean(dicom.pixelQuality)
  const signalStatus: CriterionStatus = !dicom.pixelDataPresent
    ? 'fail'
    : !dicom.pixelQuality
      ? 'warning'
      : dicom.pixelQuality.dynamicRangeFraction < 0.35 || dicom.pixelQuality.clippedFraction > 0.12
        ? 'warning'
        : 'pass'

  const signalDetail = !dicom.pixelDataPresent
    ? t('Тег Pixel Data отсутствует: оценка изображения невозможна.', 'Pixel Data is missing; image assessment is unavailable.')
    : !supportedPixels
      ? t('Метаданные прочитаны, но pixel preview недоступен для данного transfer syntax или формата пикселей.', 'Metadata was parsed, but pixel preview is unavailable for this transfer syntax or pixel format.')
      : signalStatus === 'pass'
        ? t('Диапазон яркости и доля предельных пикселей прошли технический pre-screening.', 'Intensity range and clipped-pixel rate passed technical pre-screening.')
        : t('Диапазон яркости или доля предельных пикселей требуют проверки оператором.', 'Intensity range or clipped-pixel rate requires operator review.')

  const criteria = [
    criterion(
      'dicom-integrity',
      t('Целостность DICOM', 'DICOM integrity'),
      integrityStatus === 'pass'
        ? t('Структура файла, геометрия кадра и Pixel Data прочитаны.', 'File structure, frame geometry, and Pixel Data were parsed.')
        : t('Не хватает геометрии кадра или Pixel Data.', 'Frame geometry or Pixel Data is missing.'),
      integrityStatus,
      100,
    ),
    criterion('signal', t('Техническое качество сигнала', 'Technical signal quality'), signalDetail, signalStatus, dicom.pixelQuality ? 82 : 100),
    criterion(
      'protocol',
      t('Протокол и анатомическая область', 'Protocol and anatomy'),
      protocolRecognized
        ? t('Область исследования определена по DICOM-атрибутам.', 'Study region was inferred from DICOM attributes.')
        : t('Область не указана однозначно. Перед анализом выберите протокол вручную.', 'Study region is ambiguous. Select the protocol manually before analysis.'),
      protocolRecognized ? 'pass' : 'warning',
      protocolRecognized ? 94 : 100,
    ),
    criterion(
      'anatomical-qc',
      t('Позиционирование и разметка', 'Positioning and markup'),
      t('Требуется продукционная сегментационная модель. Технический pre-screening не подменяет анатомическую оценку.', 'A production segmentation model is required. Technical pre-screening does not replace anatomical assessment.'),
      'warning',
      100,
    ),
  ]
  if (!dicom.trainingEligible) criteria.unshift(criterion(
    'training-eligibility',
    t('Пригодность для ML', 'ML eligibility'),
    t('Файл является цветным печатным представлением. Он сохранён в реестре исключений и не входит в обучающую выборку.', 'This is a color presentation render. It is recorded as excluded and does not enter the training dataset.'),
    'fail',
    100,
  ))
  return criteria
}

const summarizeStatus = (criteria: Criterion[]): StudyStatus => {
  if (criteria.some((item) => item.status === 'fail')) return 'rejected'
  if (criteria.some((item) => item.status === 'warning')) return 'review'
  return 'passed'
}

const summarizeScore = (criteria: Criterion[]) => {
  const weights: Record<CriterionStatus, number> = { pass: 100, warning: 65, fail: 20 }
  return Math.round(criteria.reduce((sum, item) => sum + weights[item.status], 0) / criteria.length)
}

export async function analyzeDicom(file: File): Promise<Study> {
  if (file.size === 0) throw new DicomAnalysisError('empty-file', 'The DICOM file is empty.')
  if (file.size > MAX_DICOM_BYTES) throw new DicomAnalysisError('file-too-large', 'The DICOM file exceeds the size limit.')

  let dicom: ParsedDicom
  try {
    dicom = parseDicomBytes(new Uint8Array(await file.arrayBuffer()))
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error)
    throw new DicomAnalysisError('invalid-dicom', `Unable to parse DICOM: ${reason}`)
  }

  const inferred = inferStudyType(dicom, file.name)
  const identitySeed = dicom.seriesInstanceUid || dicom.studyInstanceUid || `${file.name}:${file.size}`
  const patientSeed = dicom.patientId || dicom.studyInstanceUid || identitySeed
  const [identifier, patientIdentifier, accessionIdentifier, seriesIdentifier, studyIdentifier] = await Promise.all([
    stableHash(identitySeed),
    stableHash(patientSeed),
    stableHash(dicom.accessionNumber || identitySeed),
    stableHash(dicom.seriesInstanceUid || identitySeed),
    stableHash(dicom.studyInstanceUid || identitySeed),
  ])
  const criteria = buildCriteria(dicom, inferred.recognized)
  const status = summarizeStatus(criteria)
  const warnings: LocalizedText[] = [
    t('Выполнен только технический pre-screening; анатомическая модель не подключена.', 'Only technical pre-screening was performed; the anatomical model is not connected.'),
  ]
  if (!SUPPORTED_UNCOMPRESSED_TRANSFER_SYNTAXES.has(dicom.transferSyntaxUid)) {
    warnings.push(t('Сжатый pixel stream не декодирован в браузерном прототипе.', 'The compressed pixel stream was not decoded in this browser prototype.'))
  }
  if (dicom.parserWarnings.length > 0) {
    warnings.push(t(`Парсер DICOM сообщил предупреждений: ${dicom.parserWarnings.length}.`, `DICOM parser warnings: ${dicom.parserWarnings.length}.`))
  }
  if (!dicom.patientIdentityRemoved) {
    warnings.push(t('DICOM не подтверждает удаление идентификаторов пациента (0012,0062). Идентификаторы скрыты в интерфейсе.', 'DICOM does not confirm patient identity removal (0012,0062). Identifiers are masked in the interface.'))
  }
  if (dicom.burnedInAnnotation !== 'NO') {
    warnings.push(t('Отсутствие персональных данных в пикселях не подтверждено.', 'The absence of identifying text in Pixel Data is not confirmed.'))
  }
  if (!dicom.trainingEligible) {
    warnings.unshift(t('Печатный RGB DICOM исключён из ML-выборки; используйте исходную P/R-пару.', 'The presentation RGB DICOM was excluded from ML; use the source P/R pair.'))
  }

  return {
    id: `ST-${identifier}`,
    patientId: `P-${patientIdentifier}`,
    filename: `DICOM-${identifier}.dcm`,
    acquiredAt: dicom.patientIdentityRemoved ? formatDicomDate(dicom.studyDate, dicom.studyTime) : '—',
    type: inferred.type,
    status,
    score: summarizeScore(criteria),
    confidence: Math.round(criteria.reduce((sum, item) => sum + item.confidence, 0) / criteria.length),
    device: [dicom.manufacturer, dicom.modelName].filter(Boolean).join(' ') || 'Не указано',
    institution: t(dicom.institution ? 'Скрыто' : 'Не указано', dicom.institution ? 'Masked' : 'Not provided'),
    operator: dicom.operator ? 'MASKED' : '—',
    accessionNumber: `ACC-${accessionIdentifier}`,
    seriesUid: `UID-${seriesIdentifier}`,
    studyInstanceUid: dicom.studyInstanceUid ? `UID-${studyIdentifier}` : undefined,
    technical: {
      modality: dicom.modality,
      rows: dicom.rows,
      columns: dicom.columns,
      pixelSpacing: dicom.pixelSpacing ? `${dicom.pixelSpacing} mm` : undefined,
      photometricInterpretation: dicom.photometricInterpretation || undefined,
      transferSyntaxUid: dicom.transferSyntaxUid,
      bitsAllocated: dicom.bitsAllocated,
      sopClassUid: dicom.sopClassUid,
      samplesPerPixel: dicom.samplesPerPixel,
      trainingEligible: dicom.trainingEligible,
    },
    provenance: {
      mode: 'technical-screening',
      modelVersion: ANALYSIS_VERSION,
      criteriaVersion: CRITERIA_VERSION,
      processedAt: new Date().toISOString(),
      warnings,
    },
    privacy: {
      deidentificationVerified: dicom.patientIdentityRemoved,
      burnedInAnnotation: dicom.burnedInAnnotation,
    },
    previewUrl: dicom.previewUrl,
    criteria,
    recommendation: !dicom.trainingEligible
      ? t('Не использовать этот файл для обучения. Загрузите исходную P/R-пару из рабочей станции Hologic.', 'Do not use this file for training. Load the source Hologic P/R pair instead.')
      : status === 'rejected'
      ? t('Исправьте технические ошибки файла и повторите экспорт из рабочей станции.', 'Fix the technical file errors and export the study again from the workstation.')
      : t('Перед клиническим решением выполните экспертную проверку позиционирования и ROI.', 'Review positioning and ROIs before any clinical decision.'),
  }
}
