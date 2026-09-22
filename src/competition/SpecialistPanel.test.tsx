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
  expect(screen.getByText(/не меняет основное заключение/)).toBeInTheDocument()
  expect(screen.getByText(/недостаточно данных для оценки/)).toBeInTheDocument()
  expect(screen.queryByText(/Область интереса бедра/)).not.toBeInTheDocument()
})

it('does not present a failure to run as a quality verdict', () => {
  render(<SpecialistPanel result={{ mode: 'shadow', affects_decision: false, status: 'unavailable' }} />)
  expect(screen.getByText(/Дополнительную оценку получить не удалось/)).toBeInTheDocument()
  expect(screen.queryByText(/нарушение не обнаружено/)).not.toBeInTheDocument()
})
