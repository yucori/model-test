from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum


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
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GOOGLE = "google"


class LLMModelInfo(BaseModel):
    id: str
    name: str
    provider: LLMProvider
    available: bool = True
    description: str = ""


# ── Documents ────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    uploaded_at: datetime
    # Maps embedding_model_id → chunk_count.
    # Empty = not indexed by any embedding model yet.
    processed_embeddings: dict[str, int] = {}
    # True while background vectorization is in progress.
    processing: bool = False
    # Error message if text extraction (pre-embedding) failed.
    processing_error: Optional[str] = None
    # Per-embedding-model error messages (model_id → error string).
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


# ── Test run ─────────────────────────────────────────────────────────────────

class TestRunConfig(BaseModel):
    """
    Defines a test as a cartesian product:
      (embedding_model_ids × llm_model_ids) × questions

    When rag_enabled=False, embedding_model_ids is ignored and each question
    is sent to every LLM without any retrieved context.
    """
    embedding_model_ids: list[str]   # which embedding models to benchmark
    llm_model_ids: list[str]         # which LLM models to benchmark
    question_ids: Optional[list[str]] = None   # None = all questions
    rag_enabled: bool = True
    top_k: int = 3
    # Minimum cosine similarity (0–1) for a retrieved chunk to be included in context.
    # 0.0 = no filtering (include all top-k results regardless of quality).
    # Recommended: 0.3 — drops clearly irrelevant chunks before they reach the LLM.
    similarity_threshold: float = 0.0
    judge_model: str = "claude-haiku-4-5-20251001"
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
    overall: float
    reasoning: str


class TestResult(BaseModel):
    id: str
    run_id: str
    # Which embedding model retrieved the context (None when RAG disabled)
    embedding_model_id: Optional[str]
    # Which LLM generated the answer
    llm_model_id: str
    question_id: str
    question: str
    retrieved_context: list[str]
    # Cosine similarity score (0–1) for each retrieved chunk, in the same order.
    # Empty when RAG is disabled or retrieval failed.
    retrieval_scores: list[float] = []
    response: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    scores: Optional[EvaluationScores] = None
    error: Optional[str] = None
    completed_at: datetime


# ── Comparison / reporting ───────────────────────────────────────────────────

class PairSummary(BaseModel):
    """Summary for one (embedding, LLM) combination."""
    embedding_model_id: Optional[str]  # None when RAG disabled
    llm_model_id: str
    pair_label: str                    # human-readable e.g. "MiniLM + Claude Sonnet"
    avg_latency_ms: float
    avg_relevance: float
    avg_accuracy: float
    avg_helpfulness: float
    avg_korean_fluency: float
    avg_overall: float
    total_tests: int
    failed_tests: int
    avg_completion_tokens: float
    # Average cosine similarity of retrieved chunks across all questions.
    # 0.0 when RAG is disabled or no chunks passed the similarity threshold.
    avg_retrieval_score: float = 0.0


class RunComparison(BaseModel):
    run_id: str
    run_name: str
    # All (embedding × LLM) pairs, sorted by avg_overall desc
    pair_summaries: list[PairSummary]
    # LLM-level average (across all embedding models)
    llm_avg: dict[str, float]
    # Embedding-level average (across all LLM models); empty when RAG disabled
    emb_avg: dict[str, float]
    # category → pair_label → avg_overall
    by_category: dict[str, dict[str, float]]
