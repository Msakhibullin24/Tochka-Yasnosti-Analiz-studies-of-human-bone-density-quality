import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import CompetitionDynamics from './CompetitionDynamics'

afterEach(() => vi.unstubAllGlobals())

it('requires an explicit visit match and describes the comparison without false significance', async () => {
  const fetchMock = vi.fn(async () => ({ ok: true, json: async () => ({
    processing_status: 'Success', verdict: 'review', verdict_ru: 'Сопоставимость требует проверки специалистом',
    anatomical_region: 'Поясничный отдел', tolerances_clinically_validated: false,
    checks: [{ id: 'quality', label: 'Качество', status: 'review', detail: 'укладка требует проверки' }],
    bmd_change: { baseline_bmd: 0.9, followup_bmd: 0.84, change_percent: -6.7, lsc_percent: 3,
      interpretation: 'Изменение значимо (подтвердить после проверки)' },
  }) }))
  vi.stubGlobal('fetch', fetchMock)
  render(<CompetitionDynamics />)
  const baseline = new File(['first'], 'first.dcm', { type: 'application/dicom' })
  const followup = new File(['second'], 'second.dcm', { type: 'application/dicom' })
  fireEvent.change(screen.getByLabelText('Исходное исследование'), { target: { files: [baseline] } })
  fireEvent.change(screen.getByLabelText('Повторное исследование'), { target: { files: [followup] } })
  fireEvent.change(screen.getByLabelText('МПК исходная, g/cm²'), { target: { value: '0.9' } })
  fireEvent.change(screen.getByLabelText('МПК повторная, g/cm²'), { target: { value: '0.84' } })
  fireEvent.change(screen.getByLabelText('LSC учреждения, %'), { target: { value: '3' } })
  fireEvent.submit(screen.getByRole('button', { name: 'Сравнить исследования' }).closest('form')!)
  expect(screen.getByRole('alert')).toHaveTextContent('сопоставление документов')
  expect(fetchMock).not.toHaveBeenCalled()
  fireEvent.click(screen.getByLabelText(/По документам проверено/))
  fireEvent.submit(screen.getByRole('button', { name: 'Сравнить исследования' }).closest('form')!)
  expect(await screen.findByRole('heading', { name: 'Сопоставимость требует проверки специалистом' })).toBeInTheDocument()
  expect(screen.getByText(/Значимость изменения не интерпретируется/)).toBeInTheDocument()
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
  const [, options] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
  const body = options.body as FormData
  expect(body.get('baseline')).toBe(baseline)
  expect(body.get('followup')).toBe(followup)
  expect(body.get('lsc_percent')).toBe('3')
})
