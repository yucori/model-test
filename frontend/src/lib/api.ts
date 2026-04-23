import axios from 'axios'
import type {
  ChunkCompareResult,
  ChunkVariantConfig,
  DocumentInfo,
  EmbCompareResult,
  EmbTestCase,
  EmbeddingModelInfo,
  LLMModelInfo,
  ParserInfo,
  RunComparison,
  SearchCompareResult,
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
  parserId = 'pdfplumber',
  chunkSize = 600,
  overlap = 100,
  chunkStrategy = 'paragraph',
  pdfParserId?: string,
  docxParserId = 'python-docx',
) =>
  api.post('/documents/vectorize', {
    embedding_model_ids: embeddingModelIds,
    doc_ids: docIds ?? null,
    parser_id: parserId,
    pdf_parser_id: pdfParserId ?? parserId,
    docx_parser_id: docxParserId,
    chunk_size: chunkSize,
    overlap,
    chunk_strategy: chunkStrategy,
  }).then((r) => r.data)

export type DocumentTextResult = {
  doc_id: string
  filename: string
  parser: string
  chunk_size: number
  overlap: number
  char_count: number
  chunk_count: number
  avg_chunk_size: number
  min_chunk_size: number
  max_chunk_size: number
  text_preview: string
  chunks: { index: number; text: string; char_count: number }[]
}

export const getDocumentText = (
  id: string,
  parser = 'pdfplumber',
  chunkSize = 600,
  overlap = 100,
) =>
  api.get<DocumentTextResult>(`/documents/${id}/text`, {
    params: { parser, chunk_size: chunkSize, overlap },
  }).then((r) => r.data)

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

// ── Parser comparison ────────────────────────────────────────────────────────
export type ParserCompareResult = {
  parser_id: string
  elapsed_ms: number
  char_count: number
  clean_char_count: number
  word_count: number
  chunk_count: number
  korean_word_count: number
  structure_score: number
  info_density: number
  avg_chunk_size: number
  min_chunk_size: number
  max_chunk_size: number
  text_preview: string
  error: string | null
  unavailable?: boolean
  score?: number
}

export type ParserCompareResponse = {
  doc_id: string
  filename: string
  results: ParserCompareResult[]
  recommendation: { parser_id: string; score: number; reasons: string[] } | null
}

export const compareParsersFn = (docId: string, chunkSize = 600, overlap = 100) =>
  api.get<ParserCompareResponse>(`/documents/${docId}/compare-parsers`, {
    params: { chunk_size: chunkSize, overlap },
  }).then((r) => r.data)

// ── Models ────────────────────────────────────────────────────────────────────
export type ChunkStrategyInfo = {
  id: string
  name: string
  description: string
  supports_overlap: boolean
}

export const listParsers = () =>
  api.get<ParserInfo[]>('/models/parsers').then((r) => r.data)

export const listChunkStrategies = () =>
  api.get<ChunkStrategyInfo[]>('/models/chunk-strategies').then((r) => r.data)

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

// ── Pipeline stage comparison ─────────────────────────────────────────────────

export const compareChunks = (payload: {
  doc_id: string
  parser: string
  variants: ChunkVariantConfig[]
}) =>
  api.post<ChunkCompareResult>('/pipeline/chunk-compare', payload).then((r) => r.data)

export const compareSearch = (payload: {
  doc_id: string
  emb_model_id: string
  queries: string[]
  strategies: string[]
  top_k?: number
  similarity_threshold?: number
}) =>
  api.post<SearchCompareResult>('/pipeline/search-compare', payload).then((r) => r.data)

export const getBm25Status = () =>
  api.get<{ indexed_docs: string[]; total: number }>('/pipeline/bm25-status').then((r) => r.data)

export type PreviewChunk = {
  rank: number
  content: string
  matched_text: string | null
  chunk_index: number
  doc_id: string
  final_score: number
  suggested_phrase: string
}

export type GeneratedTestCase = {
  id: string
  question: string
  expected_contains: string
  category: string
}

export const generateTestCases = (payload: {
  emb_model_id: string
  doc_id?: string
  num_cases?: number
  judge_model?: string
}) =>
  api.post<{ cases: GeneratedTestCase[]; total: number }>(
    '/pipeline/generate-test-cases', payload
  ).then((r) => r.data)

export const previewChunks = (query: string, embModelId: string, topK = 5) =>
  api.get<{ query: string; emb_model_id: string; chunks: PreviewChunk[] }>(
    '/pipeline/preview-chunks',
    { params: { query, emb_model_id: embModelId, top_k: topK } },
  ).then((r) => r.data)

export const compareEmbeddings = (payload: {
  test_cases: EmbTestCase[]
  emb_model_ids: string[]
  top_k?: number
}) =>
  api.post<EmbCompareResult>('/pipeline/emb-compare', payload).then((r) => r.data)

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

export type EmbedAggregate = {
  avg_score: number
  avg_top1_score: number
  score_std: number
  hit_rate: number
  hit_at_1: number
  hit_at_3: number
  mrr: number
  score_distribution: { high: number; mid: number; low: number }
  rank_scores: number[]
  avg_retrieved: number
  relevance_threshold: number
  total_queries: number
  hit_queries: number
  dynamic_threshold: number
  dynamic_hit_at_1: number
  dynamic_hit_at_3: number
  dynamic_mrr: number
}

export const debugBatchRetrieve = (payload: {
  queries: string[]
  embedding_model_ids: string[]
  top_k: number
  similarity_threshold: number
}) =>
  api.post<{
    top_k: number
    results: Record<string, Record<string, { text: string; score: number }[]>>
    aggregates: Record<string, EmbedAggregate>
  }>('/debug/batch-retrieve', payload).then((r) => r.data)

export default api
