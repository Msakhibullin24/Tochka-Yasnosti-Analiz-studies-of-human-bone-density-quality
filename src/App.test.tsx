import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'

describe('Osseo AI workspace', () => {
  it('shows the selected study and its quality criteria', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: 'Оценка качества' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Критерии качества' })).toBeInTheDocument()
    expect(screen.getAllByText('Качественно').length).toBeGreaterThan(0)
  })

  it('switches the interface to English', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Switch to English' }))
    expect(screen.getByRole('heading', { name: 'Quality assessment' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Quality criteria' })).toBeInTheDocument()
    expect(window.localStorage.getItem('osseo-locale')).toBe('en')
  })

  it('opens the DICOM upload dialog', () => {
    const { container } = render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Загрузить DICOM' }))
    expect(screen.getByRole('dialog', { name: 'Загрузить исследование' })).toBeInTheDocument()
    expect(screen.getByText('Перетащите DICOM-файл сюда')).toBeInTheDocument()
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input, { target: { files: [new File(['not a dicom'], 'image.png', { type: 'image/png' })] } })
    expect(screen.getByRole('alert')).toHaveTextContent('Выберите файл в формате .dcm или .dicom.')
    expect(input.value).toBe('')
  })

  it('supports arrow-key navigation between analysis tabs', () => {
    render(<App />)
    const analysisTab = screen.getByRole('tab', { name: 'Анализ' })
    analysisTab.focus()
    fireEvent.keyDown(analysisTab, { key: 'ArrowRight' })
    expect(screen.getByRole('tab', { name: 'Динамика' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('heading', { name: 'Анализ в динамике' })).toBeInTheDocument()
    fireEvent.keyDown(screen.getByRole('tab', { name: 'Динамика' }), { key: 'ArrowRight' })
    expect(screen.getByRole('tab', { name: 'DICOM' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('heading', { name: 'Метаданные исследования' })).toBeInTheDocument()
    expect(window.localStorage.getItem('osseo-tab')).toBe('dicom')
  })

  it('shows a significant BMD gain only for a comparable study', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('tab', { name: 'Динамика' }))
    expect(screen.getByText('Исследования сопоставимы')).toBeInTheDocument()
    expect(screen.getByText('Значимый прирост')).toBeInTheDocument()
    expect(screen.getByText('+5,8%')).toBeInTheDocument()
    expect(screen.getByText('Предполагаемый результат')).toBeInTheDocument()
  })

  it('blocks trend interpretation when positioning and ROI are not comparable', () => {
    render(<App />)
    fireEvent.click(screen.getAllByRole('button', { name: /P-80405/ })[0])
    fireEvent.click(screen.getByRole('tab', { name: 'Динамика' }))
    expect(screen.getByText('Сопоставимость требует проверки')).toBeInTheDocument()
    expect(screen.getByText('Не интерпретируется')).toBeInTheDocument()
    expect(screen.getByText(/не сообщайте потерю BMD как значимую/i)).toBeInTheDocument()
  })

  it('keeps the upload dialog open and explains a malformed DICOM', async () => {
    const { container } = render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Загрузить DICOM' }))
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input, { target: { files: [new File(['not a dicom'], 'broken.dcm', { type: 'application/dicom' })] } })
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Не удалось прочитать DICOM'))
    expect(screen.getByRole('dialog', { name: 'Загрузить исследование' })).toBeInTheDocument()
  })
})
