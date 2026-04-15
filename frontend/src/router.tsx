import { createBrowserRouter } from 'react-router-dom'
import Layout from './components/layout/Layout'
import SetupPage from './pages/SetupPage'
import RunTestPage from './pages/RunTestPage'
import ResultsPage from './pages/ResultsPage'
import ResultDetailPage from './pages/ResultDetailPage'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <SetupPage /> },
      { path: 'run', element: <RunTestPage /> },
      { path: 'results', element: <ResultsPage /> },
      { path: 'results/:runId', element: <ResultDetailPage /> },
    ],
  },
])
