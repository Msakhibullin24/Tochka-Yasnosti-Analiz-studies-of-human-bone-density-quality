import { lazy, Suspense } from 'react'
import CompetitionWorkspace from './CompetitionWorkspace'

const ResearchWorkspace = lazy(() => import('./App'))

export default function WorkspaceRouter() {
  return <Suspense fallback={<p>Загрузка рабочего места…</p>}>
    {import.meta.env.VITE_WORKSPACE === 'research' ? <ResearchWorkspace /> : <CompetitionWorkspace />}
  </Suspense>
}
