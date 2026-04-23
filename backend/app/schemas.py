from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum


# ── Document parsers ─────────────────────────────────────────────────────────

class ParserInfo(BaseModel):
    id: str
    name: str
    available: bool
    description: str
    file_types: list[str] = []


# ── Embedding models ────────────────────────────────────────────────────────

class EmbeddingModelInfo(BaseModel):
    id: str            # e.g. "local:all-MiniLM-L6-v2"
    name: str
    provider: str      # "local" | "openai"
    available: bool
    dimensions: int
    description: str = ""


# ── LLM (generation) models ─────────────────────────────────────────────────

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


class LLMModelInfo(BaseModel):
    id: str
    name: str
    provider: LLMProvider
    available: bool = True
    description: str = ""


# ── Search strategies ────────────────────────────────────────────────────────

class SearchStrategy(str, Enum):
    SEMANTIC = "semantic"        # vector similarity only
    BM25 = "bm25"               # keyword BM25 only
    HYBRID_RRF = "hybrid_rrf"   # BM25 + semantic fused with RRF


# ── Documents ────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    uploaded_at: datetime
    processed_embeddings: dict[str, int] = {}
    processing: bool = False
    processing_error: Optional[str] = None
    embed_errors: dict[str, str] = {}


# ── Test suite ────────────────────────────────────────────────────────────────

class TestQuestion(BaseModel):
    id: str
    category: str
    question: str
    expected_topics: list[str] = []
    reference_answer: Optional[str] = None


class CreateTestQuestion(BaseModel):
    category: str
    question: str
    expected_topics: list[str] = []
    reference_answer: Optional[str] = None


class UpdateTestQuestion(BaseModel):
    category: Optional[str] = None
    question: Optional[str] = None
    expected_topics: Optional[list[str]] = None
    reference_answer: Optional[str] = None


# ── Retrieved chunk (RAG evidence) ──────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A single retrieved chunk with all its scores.

    - content: text shown to LLM (parent text when parent-child strategy is used)
    - matched_text: the child chunk that was actually matched (may differ from content)
    - semantic_score: cosine similarity 0-1, None if not used
    - bm25_score: BM25 relevance score (unnormalized), None if not used
    - final_score: the score used for final ranking (semantic, bm25, or RRF)
    """
    content: str
    matched_text: Optional[str] = None  # child chunk text (parent-child only)
    chunk_index: int
    doc_id: str
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    final_score: float


# ── Test run ─────────────────────────────────────────────────────────────────

class TestRunConfig(BaseModel):
    """
    Defines a test as a cartesian product:
      (embedding_model_ids × llm_model_ids) × questions

    When rag_enabled=False, embedding_model_ids is ignored and each question
    is sent to every LLM without any retrieved context.
    """
    embedding_model_ids: list[str]
    llm_model_ids: list[str]
    question_ids: Optional[list[str]] = None
    rag_enabled: bool = True
    top_k: int = 3
    similarity_threshold: float = 0.0
    search_strategy: SearchStrategy = SearchStrategy.SEMANTIC
    judge_model: str = ""
    run_name: Optional[str] = None


class TestRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TestRun(BaseModel):
    id: str
    name: str
    config: TestRunConfig
    status: TestRunStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_tests: int = 0
    completed_tests: int = 0
    error: Optional[str] = None


# ── Test results ─────────────────────────────────────────────────────────────

class EvaluationScores(BaseModel):
    relevance: float
    accuracy: float
    helpfulness: float
    korean_fluency: float
    source_citation: float
    overall: float
    reasoning: str


class TestResult(BaseModel):
    id: str
    run_id: str
    embedding_model_id: Optional[str]
    llm_model_id: str
    question_id: str
    question: str
    # Detailed retrieval evidence (replaces retrieved_context + retrieval_scores)
    retrieved_chunks: list[RetrievedChunk] = []
    response: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    scores: Optional[EvaluationScores] = None
    error: Optional[str] = None
    completed_at: datetime


# ── Comparison / reporting ───────────────────────────────────────────────────

class PairSummary(BaseModel):
    embedding_model_id: Optional[str]
    llm_model_id: str
    pair_label: str
    avg_latency_ms: float
    avg_relevance: float
    avg_accuracy: float
    avg_helpfulness: float
    avg_korean_fluency: float
    avg_source_citation: float
    avg_overall: float
    total_tests: int
    failed_tests: int
    avg_completion_tokens: float
    avg_retrieval_score: float = 0.0


class EmbDetail(BaseModel):
    avg_relevance: float
    avg_accuracy: float
    avg_helpfulness: float
    avg_korean_fluency: float
    avg_source_citation: float
    avg_overall: float
    avg_retrieval_score: float
    llm_count: int


class RunComparison(BaseModel):
    run_id: str
    run_name: str
    pair_summaries: list[PairSummary]
    llm_avg: dict[str, float]
    emb_avg: dict[str, float]
    emb_detail: dict[str, EmbDetail] = {}
    by_category: dict[str, dict[str, float]]


# ── Pipeline stage comparison schemas ────────────────────────────────────────

class ChunkVariantConfig(BaseModel):
    strategy: str       # paragraph | sentence | fixed | semantic | parent_child
    chunk_size: int
    overlap: int = 0
    label: str          # human-readable name


class ChunkVariantStats(BaseModel):
    config: ChunkVariantConfig
    chunk_count: int
    avg_size: float
    median_size: float
    min_size: int
    max_size: int
    size_buckets: dict[str, int]   # "< 200", "200-400", "400-600", "> 600"
    structure_aligned_count: int   # chunks starting at ## / ### heading boundary
    sample_chunks: list[str]       # first 5 chunks


class ChunkCompareResult(BaseModel):
    doc_id: str
    filename: str
    parser: str
    variants: list[ChunkVariantStats]


class SearchChunkResult(BaseModel):
    """One retrieved chunk in a search comparison result."""
    content: str
    matched_text: Optional[str] = None
    chunk_index: int
    doc_id: str
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None
    final_score: float


class SearchQueryResult(BaseModel):
    """Results for one query under one search strategy."""
    strategy: str
    label: str
    query: str
    chunks: list[SearchChunkResult]
    avg_score: float


class SearchCompareResult(BaseModel):
    doc_id: str
    emb_model_id: str
    query_results: list[SearchQueryResult]   # one per (strategy, query)
    strategies_tested: list[str]
    queries_tested: list[str]


# ── Embedding model comparison ────────────────────────────────────────────────

class EmbTestCase(BaseModel):
    id: str
    query: str
    expected_contains: str   # 이 문자열이 검색 결과 청크에 포함되면 정답
    category: str = ""


class EmbTestResult(BaseModel):
    test_case_id: str
    query: str
    expected_contains: str
    hit_rank: Optional[int] = None   # 1-indexed, None = top_k 안에 없음
    chunks: list[SearchChunkResult]


class EmbModelResult(BaseModel):
    emb_model_id: str
    test_results: list[EmbTestResult]
    hit_at_1: float
    hit_at_3: float
    hit_at_k: float
    available: bool = True   # 벡터화 안 된 경우 False


class EmbCompareResult(BaseModel):
    model_results: list[EmbModelResult]
    top_k: int
    total_cases: int
