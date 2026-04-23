import { createBrowserRouter } from 'react-router-dom'
import Layout from './components/layout/Layout'
import PipelinePage from './pages/PipelinePage'
import ResultsPage from './pages/ResultsPage'
import ResultDetailPage from './pages/ResultDetailPage'
import EmbeddingEvalPage from './pages/EmbeddingEvalPage'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <PipelinePage /> },
      { path: 'embedding-eval', element: <EmbeddingEvalPage /> },
      { path: 'results', element: <ResultsPage /> },
      { path: 'results/:runId', element: <ResultDetailPage /> },
    ],
  },
])
