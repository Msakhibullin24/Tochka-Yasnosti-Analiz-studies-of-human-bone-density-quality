import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/manrope/index.css'
import WorkspaceRouter from './WorkspaceRouter'
import './styles.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <WorkspaceRouter />
  </StrictMode>,
)
