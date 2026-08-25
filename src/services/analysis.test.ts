import { beforeAll, describe, expect, it, vi } from 'vitest'
import { analyzeDicom, isDicomFile } from './analysis'
import { parseDicomBytes } from './dicom'

const concat = (...arrays: Uint8Array[]) => {
  const result = new Uint8Array(arrays.reduce((sum, array) => sum + array.length, 0))
  let offset = 0
  for (const array of arrays) { result.set(array, offset); offset += array.length }
  return result
}

const ushort = (value: number) => new Uint8Array([value & 0xff, (value >> 8) & 0xff])
const ulong = (value: number) => new Uint8Array([value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, (value >> 24) & 0xff])

function element(group: number, id: number, vr: string, payload: Uint8Array) {
  const tag = concat(ushort(group), ushort(id), new TextEncoder().encode(vr))
  return ['OB', 'OD', 'OF', 'OL', 'OV', 'OW', 'SQ', 'SV', 'UC', 'UN', 'UR', 'UT', 'UV'].includes(vr)
    ? concat(tag, new Uint8Array(2), ulong(payload.length), payload)
    : concat(tag, ushort(payload.length), payload)
}

function textElement(group: number, id: number, vr: string, input: string) {
  const pad = vr === 'UI' ? 0 : 0x20
  const raw = new TextEncoder().encode(input)
  return element(group, id, vr, raw.length % 2 === 0 ? raw : concat(raw, new Uint8Array([pad])))
}

function makeDicom() {
  const transferSyntax = textElement(0x0002, 0x0010, 'UI', '1.2.840.10008.1.2.1')
  const metaLength = element(0x0002, 0x0000, 'UL', ulong(transferSyntax.length))
  const pixels = new Uint8Array(200)
  for (let index = 0; index < 100; index += 1) pixels.set(ushort(index * 40), index * 2)
  const dataSet = concat(
    textElement(0x0008, 0x0020, 'DA', '20260825'),
    textElement(0x0008, 0x0030, 'TM', '104200'),
    textElement(0x0008, 0x0060, 'CS', 'DX'),
    textElement(0x0010, 0x0020, 'LO', 'SECRET-PATIENT-ID'),
    textElement(0x0018, 0x0015, 'CS', 'LSPINE'),
    textElement(0x0020, 0x000d, 'UI', '1.2.826.0.1.3680043.10.1000.1'),
    textElement(0x0020, 0x000e, 'UI', '1.2.826.0.1.3680043.10.1000.2'),
    element(0x0028, 0x0002, 'US', ushort(1)),
    textElement(0x0028, 0x0004, 'CS', 'MONOCHROME2'),
    element(0x0028, 0x0010, 'US', ushort(10)),
    element(0x0028, 0x0011, 'US', ushort(10)),
    element(0x0028, 0x0100, 'US', ushort(16)),
    element(0x0028, 0x0101, 'US', ushort(12)),
    element(0x0028, 0x0102, 'US', ushort(11)),
    element(0x0028, 0x0103, 'US', ushort(0)),
    element(0x7fe0, 0x0010, 'OW', pixels),
  )
  const preamble = new Uint8Array(132)
  preamble.set(new TextEncoder().encode('DICM'), 128)
  return concat(preamble, metaLength, transferSyntax, dataSet)
}

beforeAll(() => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
})

describe('DICOM ingestion', () => {
  it('parses technical metadata and pixels from an explicit little-endian file', () => {
    const parsed = parseDicomBytes(makeDicom())
    expect(parsed.rows).toBe(10)
    expect(parsed.columns).toBe(10)
    expect(parsed.pixelDataPresent).toBe(true)
    expect(parsed.pixelQuality?.dynamicRangeFraction).toBeGreaterThan(0.9)
  })

  it('creates stable pseudonymous IDs and labels technical screening honestly', async () => {
    const bytes = makeDicom()
    const study = await analyzeDicom(new File([bytes], 'lumbar.dcm', { type: 'application/dicom' }))
    expect(study.patientId).toMatch(/^P-[0-9A-F]{8}$/)
    expect(study.patientId).not.toContain('SECRET')
    expect(study.type).toBe('spine')
    expect(study.technical.rows).toBe(10)
    expect(study.provenance.mode).toBe('technical-screening')
    expect(study.status).toBe('review')
  })

  it('rejects empty and malformed files with recoverable error codes', async () => {
    await expect(analyzeDicom(new File([], 'empty.dcm'))).rejects.toMatchObject({ code: 'empty-file' })
    await expect(analyzeDicom(new File(['not dicom'], 'broken.dcm'))).rejects.toMatchObject({ code: 'invalid-dicom' })
  })

  it('accepts the DICOM MIME type even without a conventional extension', () => {
    expect(isDicomFile(new File(['data'], 'study.bin', { type: 'application/dicom' }))).toBe(true)
  })
})
