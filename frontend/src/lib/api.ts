import axios from 'axios'
import type {
  DocumentInfo,
  EmbeddingModelInfo,
  LLMModelInfo,
  RunComparison,
  TestQuestion,
  TestResult,
  TestRun,
  TestRunConfig,
} from '../types'

const api = axios.create({ baseURL: '/api' })

// ── Documents ─────────────────────────────────────────────────────────────────
export const uploadDocument = (file: File) => {
  const form = new FormData()
  form.append('file', file)
  return api.post<DocumentInfo>('/documents', form).then((r) => r.data)
}
export const listDocuments = () =>
  api.get<DocumentInfo[]>('/documents').then((r) => r.data)

export const vectorizeDocuments = (
  embeddingModelIds: string[],
  docIds?: string[],
  chunkSize = 600,
  overlap = 100,
) =>
  api.post('/documents/vectorize', {
    embedding_model_ids: embeddingModelIds,
    doc_ids: docIds ?? null,
    chunk_size: chunkSize,
    overlap,
  }).then((r) => r.data)

export const getDocumentText = (id: string, chunkSize = 600, overlap = 100) =>
  api.get<{
    doc_id: string
    filename: string
    chunk_size: number
    overlap: number
    char_count: number
    chunk_count: number
    avg_chunk_size: number
    min_chunk_size: number
    max_chunk_size: number
    text_preview: string
    chunks: { index: number; text: string; char_count: number }[]
  }>(`/documents/${id}/text`, { params: { chunk_size: chunkSize, overlap } }).then((r) => r.data)

export const deleteDocument = (id: string) =>
  api.delete(`/documents/${id}`).then((r) => r.data)

export const getDocumentStats = () =>
  api.get('/documents/stats').then((r) => r.data)

// ── Test suite ────────────────────────────────────────────────────────────────
export const listQuestions = (category?: string) =>
  api
    .get<TestQuestion[]>('/test-suite', { params: category ? { category } : {} })
    .then((r) => r.data)
export const createQuestion = (data: Omit<TestQuestion, 'id'>) =>
  api.post<TestQuestion>('/test-suite', data).then((r) => r.data)
export const updateQuestion = (id: string, data: Partial<TestQuestion>) =>
  api.put<TestQuestion>(`/test-suite/${id}`, data).then((r) => r.data)
export const deleteQuestion = (id: string) =>
  api.delete(`/test-suite/${id}`).then((r) => r.data)
export const listCategories = () =>
  api.get<{ categories: string[] }>('/test-suite/categories').then((r) => r.data)

// ── Models ────────────────────────────────────────────────────────────────────
export const listEmbeddingModels = () =>
  api.get<EmbeddingModelInfo[]>('/models/embedding').then((r) => r.data)

export const listLLMModels = () =>
  api.get<LLMModelInfo[]>('/models/llm').then((r) => r.data)

// ── Runs ──────────────────────────────────────────────────────────────────────
export const createRun = (config: TestRunConfig) =>
  api.post<TestRun>('/runs', config).then((r) => r.data)
export const listRuns = () => api.get<TestRun[]>('/runs').then((r) => r.data)
export const getRun = (id: string) =>
  api.get<TestRun>(`/runs/${id}`).then((r) => r.data)
export const getRunResults = (id: string) =>
  api.get<TestResult[]>(`/runs/${id}/results`).then((r) => r.data)
export const getRunComparison = (id: string) =>
  api.get<RunComparison>(`/runs/${id}/comparison`).then((r) => r.data)
export const deleteRun = (id: string) =>
  api.delete(`/runs/${id}`).then((r) => r.data)

// ── Debug / diagnostics ───────────────────────────────────────────────────────
export const debugRetrieve = (payload: {
  query: string
  embedding_model_ids: string[]
  top_k: number
  similarity_threshold: number
}) =>
  api.post<{
    query: string
    top_k: number
    results: Record<string, { text: string; score: number }[]>
  }>('/debug/retrieve', payload).then((r) => r.data)

export const debugBatchRetrieve = (payload: {
  queries: string[]
  embedding_model_ids: string[]
  top_k: number
  similarity_threshold: number
}) =>
  api.post<{
    top_k: number
    results: Record<string, Record<string, { text: string; score: number }[]>>
    aggregates: Record<string, {
      avg_score: number
      avg_top1_score: number
      hit_rate: number
      total_queries: number
      hit_queries: number
    }>
  }>('/debug/batch-retrieve', payload).then((r) => r.data)

export default api
