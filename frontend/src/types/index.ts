// ── Document parsers ──────────────────────────────────────────────────────────
export interface ParserInfo {
  id: string
  name: string
  available: boolean
  description: string
  file_types: string[]
}

// ── Embedding models (RAG retrieval) ──────────────────────────────────────────
export interface EmbeddingModelInfo {
  id: string
  name: string
  provider: 'local' | 'openai'
  available: boolean
  dimensions: number
  description: string
}

// ── LLM models (answer generation) ───────────────────────────────────────────
export type LLMProvider = 'ollama' | 'google' | 'openrouter'

export interface LLMModelInfo {
  id: string
  name: string
  provider: LLMProvider
  available: boolean
  description: string
}

// ── Search strategies ─────────────────────────────────────────────────────────
export type SearchStrategy = 'semantic' | 'bm25' | 'hybrid_rrf'

// ── Documents ─────────────────────────────────────────────────────────────────
export interface DocumentInfo {
  id: string
  filename: string
  file_type: string
  size_bytes: number
  uploaded_at: string
  processed_embeddings: Record<string, number>
  processing: boolean
  processing_error: string | null
  embed_errors: Record<string, string>
}

// ── Test suite ────────────────────────────────────────────────────────────────
export interface TestQuestion {
  id: string
  category: string
  question: string
  expected_topics: string[]
  reference_answer: string | null
}

// ── Retrieved chunk (RAG evidence) ───────────────────────────────────────────
export interface RetrievedChunk {
  content: string
  matched_text: string | null
  chunk_index: number
  doc_id: string
  semantic_score: number | null
  bm25_score: number | null
  final_score: number
}

// ── Test run ──────────────────────────────────────────────────────────────────
export interface TestRunConfig {
  embedding_model_ids: string[]
  llm_model_ids: string[]
  question_ids: string[] | null
  rag_enabled: boolean
  top_k: number
  similarity_threshold: number
  search_strategy: SearchStrategy
  judge_model: string
  run_name: string | null
}

export type TestRunStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface TestRun {
  id: string
  name: string
  config: TestRunConfig
  status: TestRunStatus
  created_at: string
  completed_at: string | null
  total_tests: number
  completed_tests: number
  error: string | null
}

// ── Results ───────────────────────────────────────────────────────────────────
export interface EvaluationScores {
  relevance: number
  accuracy: number
  helpfulness: number
  korean_fluency: number
  source_citation: number
  overall: number
  reasoning: string
}

export interface TestResult {
  id: string
  run_id: string
  embedding_model_id: string | null
  llm_model_id: string
  question_id: string
  question: string
  retrieved_chunks: RetrievedChunk[]
  response: string
  latency_ms: number
  prompt_tokens: number
  completion_tokens: number
  scores: EvaluationScores | null
  error: string | null
  completed_at: string
}

// ── Comparison ────────────────────────────────────────────────────────────────
export interface PairSummary {
  embedding_model_id: string | null
  llm_model_id: string
  pair_label: string
  avg_latency_ms: number
  avg_relevance: number
  avg_accuracy: number
  avg_helpfulness: number
  avg_korean_fluency: number
  avg_source_citation: number
  avg_overall: number
  total_tests: number
  failed_tests: number
  avg_completion_tokens: number
  avg_retrieval_score: number
}

export interface EmbDetail {
  avg_relevance: number
  avg_accuracy: number
  avg_helpfulness: number
  avg_korean_fluency: number
  avg_source_citation: number
  avg_overall: number
  avg_retrieval_score: number
  llm_count: number
}

export interface RunComparison {
  run_id: string
  run_name: string
  pair_summaries: PairSummary[]
  llm_avg: Record<string, number>
  emb_avg: Record<string, number>
  emb_detail: Record<string, EmbDetail>
  by_category: Record<string, Record<string, number>>
}

// ── Pipeline stage comparison ─────────────────────────────────────────────────

export interface ChunkVariantConfig {
  strategy: string
  chunk_size: number
  overlap: number
  label: string
}

export interface ChunkVariantStats {
  config: ChunkVariantConfig
  chunk_count: number
  avg_size: number
  median_size: number
  min_size: number
  max_size: number
  size_buckets: Record<string, number>
  structure_aligned_count: number
  sample_chunks: string[]
}

export interface ChunkCompareResult {
  doc_id: string
  filename: string
  parser: string
  variants: ChunkVariantStats[]
}

export interface SearchChunkResult {
  content: string
  matched_text: string | null
  chunk_index: number
  doc_id: string
  semantic_score: number | null
  bm25_score: number | null
  final_score: number
}

export interface SearchQueryResult {
  strategy: string
  label: string
  query: string
  chunks: SearchChunkResult[]
  avg_score: number
}

export interface SearchCompareResult {
  doc_id: string
  emb_model_id: string
  query_results: SearchQueryResult[]
  strategies_tested: string[]
  queries_tested: string[]
}

// ── Embedding model comparison ─────────────────────────────────────────────────

export interface EmbTestCase {
  id: string
  query: string
  expected_contains: string
  category: string
}

export interface EmbTestResult {
  test_case_id: string
  query: string
  expected_contains: string
  hit_rank: number | null
  chunks: SearchChunkResult[]
}

export interface EmbModelResult {
  emb_model_id: string
  test_results: EmbTestResult[]
  hit_at_1: number
  hit_at_3: number
  hit_at_k: number
  available: boolean
}

export interface EmbCompareResult {
  model_results: EmbModelResult[]
  top_k: number
  total_cases: number
}
