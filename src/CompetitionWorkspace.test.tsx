import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import CompetitionWorkspace from './CompetitionWorkspace'

const row = { row_id: 'abc', anatomical_region: 'Поясничный отдел позвоночника', quality_class: '0', processing_status: 'Success', path_to_file: 'study/a.dcm', violation_codes: '', violation_scores: '{"spine_axis":0.4}', criterion_thresholds: '{"spine_axis":0.3}', quality_prob: '0.2' }
function mockAPI(submission = false, requirements?: boolean) {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    let data: unknown = {}
    if (url.endsWith('/ready')) data = { ready: true }
    else if (url.endsWith('/catalog')) data = { criteria: { spine_axis: 'Наклон оси' }, regions: { spine: ['spine_axis'] } }
    else if (url.includes('/jobs?')) data = [{ id: 'one', status: 'finished', created_at: 1, total: 1, done: 1, summary: { files: 1, submission_available: submission, submission_valid: false, requirements_complete: requirements } }]
    else if (url.endsWith('/worklist')) data = [row]
    else if (url.endsWith('/reviews') && init?.method === 'POST') data = { ...JSON.parse(String(init.body)), revision: 1 }
    else if (url.endsWith('/reviews')) data = []
    else if (url.endsWith('/images/abc')) data = { width: 300, height: 300, pixel_mm_x: .6, pixel_mm_y: .6, pixel_mm_source: 'PixelSpacing', region: 'spine', geometry: { axis: [[150, 30], [150, 270]] } }
    return { ok: true, json: async () => data } as Response
  })
}
afterEach(() => { vi.unstubAllGlobals(); localStorage.clear() })
describe('competition workspace', () => {
  it('opens the latest package and keeps secondary tools collapsed', async () => {
    vi.stubGlobal('fetch', mockAPI())
    render(<CompetitionWorkspace />)
    expect(await screen.findByRole('heading', { name: 'Поясничный отдел позвоночника' })).toBeInTheDocument()
    expect(screen.getByText('Загрузить новый пакет').closest('details')).not.toHaveAttribute('open')
    expect(screen.getByText('Подробности модели, ROI и анатомия').closest('details')).not.toHaveAttribute('open')
  })
  it('loads durable history and uses the actual decision threshold', async () => {
    vi.stubGlobal('fetch', mockAPI())
    render(<CompetitionWorkspace />)
    fireEvent.click(await screen.findByRole('button', { name: /Обработано/ }))
    expect(await screen.findByText(/порог 0.300/)).toBeInTheDocument()
    expect(await screen.findByRole('img', { name: /Исходное изображение/ })).toBeInTheDocument()
    expect(screen.queryByText('Демонстрационный режим')).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'CSV по формату ТЗ' })).not.toBeInTheDocument()
  })
  it('offers strict exports for new jobs and shows failed acceptance', async () => {
    vi.stubGlobal('fetch', mockAPI(true))
    render(<CompetitionWorkspace />)
    fireEvent.click(await screen.findByRole('button', { name: /Обработано/ }))
    expect(await screen.findByRole('link', { name: 'CSV по формату ТЗ' })).toHaveAttribute('href', '/api/v1/jobs/one/submission.csv')
    expect(screen.getByText(/Таблица для сдачи требует исправлений/)).toBeInTheDocument()
  })
  it('exposes incomplete required checks separately from file format', async () => {
    vi.stubGlobal('fetch', mockAPI(true, false))
    render(<CompetitionWorkspace />)
    fireEvent.click(await screen.findByRole('button', { name: /Обработано/ }))
    expect(await screen.findByText(/Часть обязательных проверок не завершена/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Покрытие требований ТЗ' })).toHaveAttribute('href', '/api/v1/jobs/one/requirements.json')
    expect(screen.getByRole('link', { name: 'Полное время обработки' })).toHaveAttribute('href', '/api/v1/jobs/one/timing.json')
  })
  it('saves an expert revision without replacing the model result', async () => {
    const fetch = mockAPI(); vi.stubGlobal('fetch', fetch)
    render(<CompetitionWorkspace />)
    fireEvent.click(await screen.findByRole('button', { name: /Обработано/ }))
    const author = await screen.findByLabelText('Специалист')
    await screen.findByText(/порог 0.300/)
    fireEvent.change(author, { target: { value: 'Врач' } })
    fireEvent.click(screen.getByRole('button', { name: 'Подтвердить решение' }))
    await waitFor(() => expect(screen.getByText(/Версия 1 сохранена/)).toBeInTheDocument())
    const post = fetch.mock.calls.find(([, init]) => init?.method === 'POST')
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({ expected_revision: 0, author: 'Врач', status: 'confirmed', quality_class: 0 })
    expect(screen.getByText(/Балл модели: 0.2 из 1/)).toBeInTheDocument()
  })
  it('shows unavailable service rather than local demonstration predictions', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Сервис недоступен')))
    render(<CompetitionWorkspace />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Сервис недоступен')
    expect(screen.queryByText('Нарушений не выявлено')).not.toBeInTheDocument()
  })
  it('restores the open image, queue filter and zoom after remount', async () => {
    vi.stubGlobal('fetch', mockAPI())
    const view = render(<CompetitionWorkspace />)
    fireEvent.click(await screen.findByRole('button', { name: /Обработано/ }))
    await screen.findByRole('img', { name: /Исходное изображение/ })
    fireEvent.change(screen.getByLabelText('Фильтр'), { target: { value: 'pending' } })
    fireEvent.change(screen.getByLabelText('Масштаб'), { target: { value: '2' } })
    view.unmount()
    render(<CompetitionWorkspace />)
    await screen.findByRole('img', { name: /Исходное изображение/ })
    expect(screen.getByLabelText('Фильтр')).toHaveValue('pending')
    expect(screen.getByLabelText('Масштаб')).toHaveValue('2')
  })
  it('edits points by keyboard, supports undo and protects original geometry', async () => {
    vi.stubGlobal('fetch', mockAPI())
    render(<CompetitionWorkspace />)
    fireEvent.click(await screen.findByRole('button', { name: /Обработано/ }))
    await screen.findByRole('img', { name: /Исходное изображение/ })
    fireEvent.click(screen.getByLabelText('Перемещать точки'))
    const point = screen.getByRole('button', { name: 'axis, точка 1. Перемещение стрелками' })
    const before = point.getAttribute('cx')
    fireEvent.keyDown(point, { key: 'ArrowRight', shiftKey: true })
    expect(Number(point.getAttribute('cx'))).toBeCloseTo(Number(before) + 10)
    fireEvent.click(screen.getByRole('button', { name: 'Показать машинную разметку' }))
    expect(screen.queryByRole('button', { name: /axis, точка 1/ })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Показать мою разметку' }))
    expect(Number(screen.getByRole('button', { name: /axis, точка 1/ }).getAttribute('cx'))).toBeCloseTo(Number(before) + 10)
    fireEvent.click(screen.getByRole('button', { name: 'Отменить изменение' }))
    expect(screen.getByRole('button', { name: /axis, точка 1/ })).toHaveAttribute('cx', before)
  })
  it('restores annotations and comments from the browser copy after remount', async () => {
    vi.stubGlobal('fetch', mockAPI())
    const view = render(<CompetitionWorkspace />)
    fireEvent.click(await screen.findByRole('button', { name: /Обработано/ }))
    await screen.findByRole('img', { name: /Исходное изображение/ })
    fireEvent.change(screen.getByLabelText('Комментарий'), { target: { value: 'Спорная область' } })
    fireEvent.change(screen.getByLabelText('Замечание к axis'), { target: { value: 'Проверить линию' } })
    expect(await screen.findByText(/Рабочая копия сохранена/)).toBeInTheDocument()
    view.unmount()
    render(<CompetitionWorkspace />)
    fireEvent.click(await screen.findByRole('button', { name: 'Восстановить рабочую копию' }))
    expect(screen.getByLabelText('Комментарий')).toHaveValue('Спорная область')
    expect(screen.getByLabelText('Замечание к axis')).toHaveValue('Проверить линию')
  })
  it('saves a second opinion request and exposes it in the queue', async () => {
    const fetch = mockAPI(); vi.stubGlobal('fetch', fetch)
    render(<CompetitionWorkspace />)
    fireEvent.click(await screen.findByRole('button', { name: /Обработано/ }))
    await screen.findByRole('img', { name: /Исходное изображение/ })
    fireEvent.change(screen.getByLabelText('Специалист'), { target: { value: 'Врач' } })
    fireEvent.click(screen.getByRole('button', { name: 'Добавить последующее действие' }))
    fireEvent.change(screen.getByLabelText('Вопрос или задача'), { target: { value: 'Проверить ROI' } })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    await screen.findByText(/Версия 1 сохранена/)
    expect(JSON.parse(String(fetch.mock.calls.find(([, init]) => init?.method === 'POST')?.[1]?.body)).followups[0]).toMatchObject({ question: 'Проверить ROI', state: 'open', kind: 'second_opinion' })
    fireEvent.change(screen.getByLabelText('Фильтр'), { target: { value: 'second_opinion' } })
    expect(screen.getByText(/Открытых действий: 1/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Скачать отчёт версии 1' })).toHaveAttribute('href', '/api/v1/jobs/one/images/abc/reviews/1/report.html')
  })
  it('keeps edits on conflict and requires explicit adoption of the current revision', async () => {
    const normal = mockAPI()
    let conflict = false
    let posted = 0
    const fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith('/reviews') && init?.method === 'POST') {
        posted++
        if (posted === 1) { conflict = true; return { ok: false, status: 409, json: async () => ({ detail: 'conflict' }) } as Response }
        return { ok: true, json: async () => ({ ...JSON.parse(String(init.body)), revision: 3 }) } as Response
      }
      if (String(input).endsWith('/reviews') && conflict) return { ok: true, json: async () => [{ revision: 2, author: 'Другой врач', status: 'draft', quality_class: 0, violations: [], geometry: [], comment: 'Другое мнение' }] } as Response
      return normal(input, init)
    })
    vi.stubGlobal('fetch', fetch)
    render(<CompetitionWorkspace />)
    fireEvent.click(await screen.findByRole('button', { name: /Обработано/ }))
    await screen.findByRole('img', { name: /Исходное изображение/ })
    fireEvent.change(screen.getByLabelText('Специалист'), { target: { value: 'Врач' } })
    fireEvent.change(screen.getByLabelText('Комментарий'), { target: { value: 'Моя правка' } })
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    const adopt = await screen.findByRole('button', { name: 'Продолжить с моими правками после версии 2' })
    expect(screen.getByLabelText('Комментарий')).toHaveValue('Моя правка')
    fireEvent.click(adopt)
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить черновик' }))
    await screen.findByText(/Версия 3 сохранена/)
    const posts = fetch.mock.calls.filter(([, init]) => init?.method === 'POST')
    expect(JSON.parse(String(posts[1][1]?.body))).toMatchObject({ expected_revision: 2, comment: 'Моя правка' })
  })
})


it('allows drafts but explains why a positive decision without a type cannot be confirmed', async () => {
  vi.stubGlobal('fetch', mockAPI())
  render(<CompetitionWorkspace />)
  fireEvent.click(await screen.findByRole('button', { name: /Обработано/ }))
  await screen.findByRole('img', { name: /Исходное изображение/ })
  fireEvent.change(screen.getByLabelText('Качество'), { target: { value: '1' } })
  expect(screen.getByRole('button', { name: 'Подтвердить решение' })).toBeDisabled()
  expect(screen.getByRole('button', { name: 'Сохранить черновик' })).not.toBeDisabled()
  expect(screen.getByText(/Для подтверждения укажите тип нарушения/)).toBeInTheDocument()
  fireEvent.click(screen.getByRole('checkbox', { name: 'Наклон оси' }))
  expect(screen.getByRole('button', { name: 'Подтвердить решение' })).not.toBeDisabled()
})
