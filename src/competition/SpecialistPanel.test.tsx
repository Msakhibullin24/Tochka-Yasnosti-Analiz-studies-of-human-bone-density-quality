import { render, screen } from '@testing-library/react'
import { expect, it } from 'vitest'
import SpecialistPanel from './SpecialistPanel'

it('keeps unknown assessment distinct from a pass and hides inapplicable criteria', () => {
  render(<SpecialistPanel result={{ mode: 'shadow', affects_decision: false, predictions: {
    quality: { score: .8, threshold: .5, status: 'fail' },
    spine_coverage: { score: .2, threshold: null, status: 'undetermined' },
    hip_roi_coverage: { score: null, threshold: null, status: 'not_applicable' },
  } }} />)
  expect(screen.getByRole('region', { name: 'Экспериментальная оценка качества' })).toBeInTheDocument()
  expect(screen.getByText(/не меняют основной конкурсный вывод/)).toBeInTheDocument()
  expect(screen.getByText(/недостаточно данных для оценки/)).toBeInTheDocument()
  expect(screen.queryByText(/Область интереса бедра/)).not.toBeInTheDocument()
})

it('does not present a failure to run as a quality verdict', () => {
  render(<SpecialistPanel result={{ mode: 'shadow', affects_decision: false, status: 'unavailable' }} />)
  expect(screen.getByText(/Оценка этой модели недоступна/)).toBeInTheDocument()
  expect(screen.queryByText(/нарушение не обнаружено/)).not.toBeInTheDocument()
})

it('shows independent model results without turning an unavailable model into a verdict', () => {
  render(<SpecialistPanel results={{
    spine_convnext: { mode: 'shadow', affects_decision: false, status: 'ok', predictions: {
      quality: { score: .8, threshold: .5, status: 'fail' },
    } },
    hip_convnext: { mode: 'shadow', affects_decision: false, status: 'not_applicable' },
    general_efficientnet: { mode: 'shadow', affects_decision: false, status: 'unavailable' },
  }} />)
  expect(screen.getByText('ConvNeXt: позвоночник')).toBeInTheDocument()
  expect(screen.getByText('ConvNeXt: бедро')).toBeInTheDocument()
  expect(screen.getByText('EfficientNet: общая оценка')).toBeInTheDocument()
  expect(screen.getByText(/Модель предназначена для другой анатомической области/)).toBeInTheDocument()
  expect(screen.getByText(/Оценка этой модели недоступна/)).toBeInTheDocument()
})
