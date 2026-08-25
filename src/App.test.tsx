import { fireEvent, render, screen } from '@testing-library/react'
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
  })

  it('opens the DICOM upload dialog', () => {
    const { container } = render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Загрузить DICOM' }))
    expect(screen.getByRole('dialog', { name: 'Загрузить исследование' })).toBeInTheDocument()
    expect(screen.getByText('Перетащите DICOM-файл сюда')).toBeInTheDocument()
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input, { target: { files: [new File(['not a dicom'], 'image.png', { type: 'image/png' })] } })
    expect(screen.getByRole('alert')).toHaveTextContent('Выберите файл в формате .dcm или .dicom.')
  })
})
