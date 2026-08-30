import * as dicomParser from 'dicom-parser'

export const SUPPORTED_UNCOMPRESSED_TRANSFER_SYNTAXES = new Set([
  '1.2.840.10008.1.2',
  '1.2.840.10008.1.2.1',
  '1.2.840.10008.1.2.2',
])

export interface ParsedDicom {
  patientId: string
  accessionNumber: string
  studyInstanceUid: string
  seriesInstanceUid: string
  manufacturer: string
  modelName: string
  institution: string
  operator: string
  modality: string
  bodyPart: string
  description: string
  protocolName: string
  studyDate: string
  studyTime: string
  transferSyntaxUid: string
  rows?: number
  columns?: number
  pixelSpacing?: string
  photometricInterpretation?: string
  bitsAllocated?: number
  patientIdentityRemoved: boolean
  burnedInAnnotation?: 'YES' | 'NO'
  sopClassUid: string
  samplesPerPixel: number
  trainingEligible: boolean
  exclusionReason?: 'secondary-capture' | 'color-presentation'
  pixelDataPresent: boolean
  previewUrl?: string
  pixelQuality?: PixelQuality
  parserWarnings: string[]
}

export interface PixelQuality {
  dynamicRangeFraction: number
  clippedFraction: number
  p01: number
  p99: number
}

const value = (dataSet: dicomParser.DataSet, tag: string, fallback = '') =>
  (dataSet.string(tag) ?? fallback).replace(/[\u0000-\u001f\u007f]/g, '').trim().slice(0, 256)

const percentile = (sorted: number[], fraction: number) => {
  if (sorted.length === 0) return 0
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * fraction)))]
}

function readPixels(dataSet: dicomParser.DataSet, transferSyntaxUid: string) {
  const element = dataSet.elements.x7fe00010
  const rows = dataSet.uint16('x00280010')
  const columns = dataSet.uint16('x00280011')
  const samplesPerPixel = dataSet.uint16('x00280002') ?? 1
  const bitsAllocated = dataSet.uint16('x00280100')
  const bitsStored = dataSet.uint16('x00280101') ?? bitsAllocated
  const highBit = dataSet.uint16('x00280102') ?? ((bitsStored ?? 1) - 1)
  const signed = dataSet.uint16('x00280103') === 1

  if (!element || !rows || !columns || samplesPerPixel !== 1 || !bitsAllocated || !bitsStored) return undefined
  if (!SUPPORTED_UNCOMPRESSED_TRANSFER_SYNTAXES.has(transferSyntaxUid) || ![8, 16].includes(bitsAllocated)) return undefined

  const pixelCount = rows * columns
  if (pixelCount <= 0 || pixelCount > 25_000_000) return undefined
  const bytesPerPixel = bitsAllocated / 8
  if (element.length < pixelCount * bytesPerPixel) return undefined

  const dataView = new DataView(dataSet.byteArray.buffer, dataSet.byteArray.byteOffset + element.dataOffset, pixelCount * bytesPerPixel)
  const littleEndian = transferSyntaxUid !== '1.2.840.10008.1.2.2'
  const shift = Math.max(0, highBit - bitsStored + 1)
  const mask = bitsStored === 16 ? 0xffff : (1 << bitsStored) - 1
  const signBit = 1 << (bitsStored - 1)
  const signRange = 1 << bitsStored
  const slope = dataSet.floatString('x00281053') ?? 1
  const intercept = dataSet.floatString('x00281052') ?? 0
  const pixels = new Float32Array(pixelCount)

  for (let index = 0; index < pixelCount; index += 1) {
    const raw = bitsAllocated === 8 ? dataView.getUint8(index) : dataView.getUint16(index * 2, littleEndian)
    let stored = (raw >> shift) & mask
    if (signed && (stored & signBit)) stored -= signRange
    pixels[index] = stored * slope + intercept
  }
  return { pixels, rows, columns }
}

function summarizePixels(pixels: Float32Array): PixelQuality {
  const step = Math.max(1, Math.floor(pixels.length / 16_384))
  const sample: number[] = []
  for (let index = 0; index < pixels.length; index += step) sample.push(pixels[index])
  sample.sort((a, b) => a - b)
  const min = sample[0] ?? 0
  const max = sample[sample.length - 1] ?? 0
  const p01 = percentile(sample, 0.01)
  const p99 = percentile(sample, 0.99)
  const fullRange = Math.max(1, max - min)
  const edgeTolerance = fullRange * 0.001
  const clipped = sample.filter((entry) => entry <= min + edgeTolerance || entry >= max - edgeTolerance).length
  return {
    dynamicRangeFraction: Math.max(0, Math.min(1, (p99 - p01) / fullRange)),
    clippedFraction: clipped / Math.max(1, sample.length),
    p01,
    p99,
  }
}

function createPreview(pixels: Float32Array, rows: number, columns: number, quality: PixelQuality, invert: boolean) {
  if (typeof document === 'undefined') return undefined
  const maxSide = 1024
  const scale = Math.min(1, maxSide / Math.max(rows, columns))
  const outputWidth = Math.max(1, Math.round(columns * scale))
  const outputHeight = Math.max(1, Math.round(rows * scale))
  const canvas = document.createElement('canvas')
  canvas.width = outputWidth
  canvas.height = outputHeight
  const context = canvas.getContext('2d')
  if (!context) return undefined
  const image = context.createImageData(outputWidth, outputHeight)
  const windowWidth = Math.max(1, quality.p99 - quality.p01)

  for (let y = 0; y < outputHeight; y += 1) {
    const sourceY = Math.min(rows - 1, Math.floor(y / scale))
    for (let x = 0; x < outputWidth; x += 1) {
      const sourceX = Math.min(columns - 1, Math.floor(x / scale))
      const normalized = Math.max(0, Math.min(1, (pixels[sourceY * columns + sourceX] - quality.p01) / windowWidth))
      const gray = Math.round((invert ? 1 - normalized : normalized) * 255)
      const target = (y * outputWidth + x) * 4
      image.data[target] = gray
      image.data[target + 1] = gray
      image.data[target + 2] = gray
      image.data[target + 3] = 255
    }
  }
  context.putImageData(image, 0, 0)
  return canvas.toDataURL('image/png')
}

export function parseDicomBytes(bytes: Uint8Array): ParsedDicom {
  const dataSet = dicomParser.parseDicom(bytes)
  const transferSyntaxUid = value(dataSet, 'x00020010', '1.2.840.10008.1.2')
  const photometricInterpretation = value(dataSet, 'x00280004')
  const sopClassUid = value(dataSet, 'x00080016')
  const samplesPerPixel = dataSet.uint16('x00280002') ?? 1
  const colorPresentation = samplesPerPixel > 1 || /^(RGB|YBR)/i.test(photometricInterpretation)
  const decoded = readPixels(dataSet, transferSyntaxUid)
  const pixelQuality = decoded ? summarizePixels(decoded.pixels) : undefined
  const previewUrl = decoded && pixelQuality
    ? createPreview(decoded.pixels, decoded.rows, decoded.columns, pixelQuality, photometricInterpretation === 'MONOCHROME1')
    : undefined

  return {
    patientId: value(dataSet, 'x00100020'),
    accessionNumber: value(dataSet, 'x00080050'),
    studyInstanceUid: value(dataSet, 'x0020000d'),
    seriesInstanceUid: value(dataSet, 'x0020000e'),
    manufacturer: value(dataSet, 'x00080070'),
    modelName: value(dataSet, 'x00081090'),
    institution: value(dataSet, 'x00080080'),
    operator: value(dataSet, 'x00081070'),
    modality: value(dataSet, 'x00080060', 'OT'),
    bodyPart: value(dataSet, 'x00180015'),
    description: [value(dataSet, 'x00081030'), value(dataSet, 'x0008103e')].filter(Boolean).join(' · '),
    protocolName: value(dataSet, 'x00181030'),
    studyDate: value(dataSet, 'x00080020') || value(dataSet, 'x00080021'),
    studyTime: value(dataSet, 'x00080030') || value(dataSet, 'x00080031'),
    transferSyntaxUid,
    rows: dataSet.uint16('x00280010'),
    columns: dataSet.uint16('x00280011'),
    pixelSpacing: value(dataSet, 'x00280030').replace(/\\/g, ' × '),
    photometricInterpretation,
    bitsAllocated: dataSet.uint16('x00280100'),
    patientIdentityRemoved: value(dataSet, 'x00120062').toUpperCase() === 'YES',
    burnedInAnnotation: ['YES', 'NO'].includes(value(dataSet, 'x00280301').toUpperCase())
      ? value(dataSet, 'x00280301').toUpperCase() as 'YES' | 'NO'
      : undefined,
    sopClassUid,
    samplesPerPixel,
    trainingEligible: !colorPresentation,
    exclusionReason: colorPresentation
      ? sopClassUid === '1.2.840.10008.5.1.4.1.1.7' ? 'secondary-capture' : 'color-presentation'
      : undefined,
    pixelDataPresent: Boolean(dataSet.elements.x7fe00010),
    previewUrl,
    pixelQuality,
    parserWarnings: dataSet.warnings.map((warning) => String(warning).slice(0, 256)),
  }
}
