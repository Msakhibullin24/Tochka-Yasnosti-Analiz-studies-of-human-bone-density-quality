import page from '../../competition/anatomy_review/index.html?raw'
import script from '../../competition/anatomy_review/review.js?raw'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent } from '@testing-library/react'

const fixture = {
  version: 1, results_sha256: 'hash', reviewer: '', annotation_source: '', independent_of_predictions: false,
  cases: [{ image_uid: '1.2', image: 'images/0000.png', width: 100, height: 90, source_sha256: 'source', pixel_sha256: 'pixel',
    status: 'unreviewed', projection: null, landmarks: { Th12: { visible: null, point: null } } }],
}
const get = (id: string) => document.getElementById(id) as HTMLInputElement
const click = (id: string) => fireEvent.click(get(id))
function set(id: string, value: string) { get(id).value = value; fireEvent.change(get(id)) }
function visible() { fireEvent.click(document.querySelector('input[value="yes"]')!) }
function annotate() { set('projection', 'frontal'); visible(); set('x', '0'); set('y', '0'); click('point'); click('confirm') }
let saved: Blob
beforeEach(() => {
  document.documentElement.innerHTML = page.replace('__REVIEW_DATA__', JSON.stringify(fixture))
  vi.stubGlobal('structuredClone', (value: unknown) => JSON.parse(JSON.stringify(value)))
  vi.stubGlobal('URL', { createObjectURL: (blob: Blob) => { saved = blob; return 'blob:local' }, revokeObjectURL: vi.fn() })
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
  window.eval(script)
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); document.body.innerHTML = '' })
async function exported() {
  const reader = new FileReader()
  const contents = new Promise<string>((resolve, reject) => { reader.onload = () => resolve(String(reader.result)); reader.onerror = reject })
  reader.readAsText(saved)
  return JSON.parse(await contents)
}
describe('blind anatomy review', () => {
  it('requires explicit visibility, both coordinates, and projection before confirmation', () => {
    click('confirm'); expect(get('state')).toHaveTextContent('Черновик')
    set('projection', 'frontal'); visible(); set('x', '0'); click('point')
    expect(get('message')).toHaveTextContent('Введите X и Y')
    set('y', '0'); click('point'); click('confirm')
    expect(get('state')).toHaveTextContent('подтверждена')
    set('x', '100'); click('point'); expect(get('message')).toHaveTextContent('внутри исходного')
  })
  it('invalidates confirmation on a changed point or invisible landmark', () => {
    annotate(); set('x', '5'); click('point'); expect(get('state')).toHaveTextContent('Черновик')
    click('confirm'); fireEvent.click(document.querySelector('input[value="no"]')!)
    expect(get('x').value).toBe(''); expect(get('state')).toHaveTextContent('Черновик')
  })
  it('exports only independently confirmed references and preserves source identity', async () => {
    annotate(); click('export'); expect(get('message')).toHaveTextContent('Укажите специалиста')
    set('reviewer', 'reviewer'); set('source', 'independent source'); fireEvent.click(get('independent')); click('export')
    const result = await exported()
    expect(result.cases[0].landmarks.Th12).toEqual({ visible: true, point: [0, 0] })
    expect(result.cases[0].source_sha256).toBe('source')
    expect(result.independent_of_predictions).toBe(true)
  })
  it('rejects an imported draft from another image without changing the current annotation', async () => {
    annotate()
    const changed = JSON.parse(JSON.stringify(fixture)); changed.cases[0].source_sha256 = 'other'
    const file = new File([JSON.stringify(changed)], 'draft.json', { type: 'application/json' })
    Object.defineProperty(file, 'text', { value: async () => JSON.stringify(changed) })
    fireEvent.change(get('import'), { target: { files: [file] } })
    await vi.waitFor(() => expect(get('message')).toHaveTextContent('не совпадают'))
    expect(get('state')).toHaveTextContent('подтверждена')
  })
})


describe('named contours and rotation', () => {
  function extendedCase(rotation = false) {
    const c = {...fixture.cases[0], named_regions: {L1: {visible: null, polygon: null}}, ...(rotation ? {rotation: null} : {})}
    document.documentElement.innerHTML = page.replace('__REVIEW_DATA__', JSON.stringify({...fixture, cases: [c]}))
    window.eval(script)
  }
  function vertex(x: number, y: number) { set('rx', String(x)); set('ry', String(y)); click('vertex') }
  it('requires a complete contour and invalidates confirmation after editing', () => {
    extendedCase(); annotate()
    expect(get('state')).toHaveTextContent('Черновик')
    set('region-visible', 'yes'); vertex(5, 5); vertex(30, 5)
    click('confirm'); expect(get('state')).toHaveTextContent('Черновик')
    vertex(30, 30); vertex(5, 30); click('confirm')
    expect(get('state')).toHaveTextContent('подтверждена')
    click('undo-vertex'); expect(get('state')).toHaveTextContent('Черновик')
  })
  it('rejects a self crossing contour', () => {
    extendedCase(); annotate(); set('region-visible', 'yes')
    vertex(5, 5); vertex(30, 30); vertex(5, 30); vertex(30, 5); click('confirm')
    expect(get('state')).toHaveTextContent('Черновик')
    expect(get('region-state')).toHaveTextContent('не завершён')
  })
  it('requires rotation explicitly and preserves the subtype in exports', async () => {
    extendedCase(true); annotate(); set('region-visible', 'no'); click('confirm')
    expect(get('state')).toHaveTextContent('Черновик')
    set('rotation', 'excessive'); click('confirm'); expect(get('state')).toHaveTextContent('подтверждена')
    set('reviewer', 'reviewer'); set('source', 'independent source'); fireEvent.click(get('independent')); click('export')
    expect((await exported()).cases[0].rotation).toBe('excessive')
  })
})


describe('independent neck axis and anatomical bone contours', () => {
  function hip() {
    const c = {...fixture.cases[0], landmarks: {femoral_neck: {visible: null, point: null}},
      named_regions: {femur: {visible: null, polygon: null}},
      axes: {femoral_neck_axis: {visible: null, points: null}}}
    document.documentElement.innerHTML = page.replace('__REVIEW_DATA__', JSON.stringify({...fixture, review_package_version: 3, cases: [c]}))
    window.eval(script)
    set('projection', 'frontal'); visible(); set('x', '5'); set('y', '5'); click('point')
  }
  function endpoint(end: string, x: string, y: string) {
    set('axis-end', end); set('ax', x); set('ay', y); click('axis-point')
  }
  it('requires two distinct axis endpoints and invalidates confirmation on edit', async () => {
    hip(); set('region-visible', 'no'); set('axis-visible', 'yes')
    endpoint('0', '5', '5'); click('confirm'); expect(get('state')).toHaveTextContent('Черновик')
    endpoint('1', '5', '5'); click('confirm'); expect(get('state')).toHaveTextContent('Черновик')
    endpoint('1', '20', '15'); click('confirm'); expect(get('state')).toHaveTextContent('подтверждена')
    set('reviewer', 'expert'); set('source', 'independent'); fireEvent.click(get('independent')); click('export')
    expect((await exported()).cases[0].axes.femoral_neck_axis.points).toEqual([[5,5],[20,15]])
    endpoint('1', '21', '15'); expect(get('state')).toHaveTextContent('Черновик')
  })
  it('accepts an anatomical concavity without converting it to a convex hull', () => {
    hip(); set('axis-visible','no'); set('region-visible','yes')
    for (const [x,y] of [[2,2],[12,2],[12,12],[8,12],[8,6],[6,6],[6,12],[2,12]]) {
      set('rx',String(x)); set('ry',String(y)); click('vertex')
    }
    click('confirm'); expect(get('state')).toHaveTextContent('подтверждена')
  })
})

describe('full Th12 body annotation', () => {
  function spine() {
    const c = {...fixture.cases[0], th12_full_body: {status: null, quadrilateral: null}}
    document.documentElement.innerHTML = page.replace('__REVIEW_DATA__', JSON.stringify({...fixture, review_package_version: 4, cases: [c]}))
    window.eval(script)
    set('projection', 'frontal'); visible(); set('x', '5'); set('y', '5'); click('point')
  }
  function corner(x: number, y: number) { set('amodal-x',String(x)); set('amodal-y',String(y)); click('amodal-point') }
  it('requires an explicit full-body assessment and accepts an off-image corner', async () => {
    spine(); click('confirm'); expect(get('state')).toHaveTextContent('Черновик')
    set('amodal-status','assessable')
    for (const [x,y] of [[5,-10],[40,-10],[40,20],[5,20]]) corner(x,y)
    click('confirm'); expect(get('state')).toHaveTextContent('подтверждена')
    set('reviewer','expert'); set('source','independent'); fireEvent.click(get('independent')); click('export')
    expect((await exported()).cases[0].th12_full_body.quadrilateral[0]).toEqual([5,-10])
    click('amodal-undo'); expect(get('state')).toHaveTextContent('Черновик')
  })
  it('allows an explicit unassessable case without guessed coordinates', () => {
    spine(); set('amodal-status','unassessable'); click('confirm')
    expect(get('state')).toHaveTextContent('подтверждена')
  })
})
