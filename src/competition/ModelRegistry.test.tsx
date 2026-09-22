import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import ModelRegistry from './ModelRegistry'

afterEach(() => vi.unstubAllGlobals())
it('loads available model inventory only on request', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ configured: true, capabilities: [
    { id: 'yolo26x', readiness: 'requires_target_training', artifacts_verified: true },
  ] }) })
  vi.stubGlobal('fetch', fetch)
  render(<ModelRegistry />)
  expect(fetch).not.toHaveBeenCalled()
  fireEvent.click(screen.getByText('Модели и данные проекта'))
  fireEvent.click(screen.getByRole('button', { name: 'Проверить файлы моделей' }))
  expect(await screen.findByText(/Для имплантов и артефактов нужна разметка/)).toBeInTheDocument()
})
it('shows a recoverable error when the inventory cannot load', async () => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))
  render(<ModelRegistry />)
  fireEvent.click(screen.getByText('Модели и данные проекта'))
  fireEvent.click(screen.getByRole('button', { name: 'Проверить файлы моделей' }))
  expect(await screen.findByText(/Не удалось проверить модели/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Проверить файлы моделей' })).toBeEnabled()
})
