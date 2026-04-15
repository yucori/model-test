// ── Embedding models (RAG retrieval) ──────────────────────────────────────────
export interface EmbeddingModelInfo {
  id: string           // e.g. "local:all-MiniLM-L6-v2"
  name: string
  provider: 'local' | 'openai' | 'ollama'
  available: boolean
  dimensions: number
  description: string
}

// ── LLM models (answer generation) ───────────────────────────────────────────
export type LLMProvider = 'anthropic' | 'openai' | 'ollama' | 'google'

export interface LLMModelInfo {
  id: string
  name: string
  provider: LLMProvider
  available: boolean
  description: string
}

// ── Documents ─────────────────────────────────────────────────────────────────
export interface DocumentInfo {
  id: string
  filename: string
  file_type: string
  size_bytes: number
  uploaded_at: string
  /** embedding_model_id → chunk_count */
  processed_embeddings: Record<string, number>
  /** True while background vectorization is running */
  processing: boolean
  processing_error: string | null
  /** Per-embedding-model error messages (model_id → error string) */
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

// ── Test run ──────────────────────────────────────────────────────────────────
export interface TestRunConfig {
  /** Embedding models to test (determines RAG retrieval) */
  embedding_model_ids: string[]
  /** LLM models to test (determines answer generation) */
  llm_model_ids: string[]
  question_ids: string[] | null
  rag_enabled: boolean
  top_k: number
  /** Minimum cosine similarity (0–1) to include a retrieved chunk. 0 = no filtering. */
  similarity_threshold: number
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
  overall: number
  reasoning: string
}

export interface TestResult {
  id: string
  run_id: string
  /** Which embedding model retrieved the context (null when RAG disabled) */
  embedding_model_id: string | null
  /** Which LLM generated the answer */
  llm_model_id: string
  question_id: string
  question: string
  retrieved_context: string[]
  /** Cosine similarity score (0–1) per retrieved chunk. Empty when RAG disabled. */
  retrieval_scores: number[]
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
  avg_overall: number
  total_tests: number
  failed_tests: number
  avg_completion_tokens: number
  /** Average cosine similarity of retrieved chunks (0–1). 0 when RAG disabled. */
  avg_retrieval_score: number
}

export interface RunComparison {
  run_id: string
  run_name: string
  /** All (embedding × LLM) pairs, sorted by avg_overall desc */
  pair_summaries: PairSummary[]
  /** LLM → avg_overall (aggregated across all embeddings) */
  llm_avg: Record<string, number>
  /** Embedding → avg_overall (aggregated across all LLMs) */
  emb_avg: Record<string, number>
  /** category → pair_label → avg_overall */
  by_category: Record<string, Record<string, number>>
}
