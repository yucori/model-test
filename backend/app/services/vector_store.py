"""
Per-embedding-model ChromaDB collections.

Each embedding model gets its own isolated collection so that:
  - Vector dimensions don't clash (MiniLM=384, bge-m3=1024, etc.)
  - We can compare how different embeddings affect RAG retrieval quality.

The embedding function is set on the collection and called automatically by
ChromaDB when add(documents=...) or query(query_texts=...) is called.
"""
import os
import re
from typing import TYPE_CHECKING, Any
import chromadb

if TYPE_CHECKING:
    from app.schemas import RetrievedChunk
from chromadb.config import Settings as ChromaSettings

from app.config import settings as app_settings
from app.schemas import EmbeddingModelInfo

# ── HuggingFace / sentence-transformers embedding function ────────────────────

# Module-level model cache: avoids reloading a model on every request.
_hf_model_cache: dict[str, Any] = {}


class _HuggingFaceEmbeddingFn:
    """ChromaDB-compatible embedding function using sentence-transformers.

    Models are loaded lazily on first call and cached for the lifetime of the
    process, so the ~GB download only happens once per model.

    Supports instruction prefixes (passage_prefix / query_prefix) for models
    that require different prompts for document indexing vs. query search:
      - bge-m3:          passage: "passage: "   query: "query: "
      - nomic-v1.5:      passage: "search_document: "  query: "search_query: "
      - qwen3-embedding: passage: ""  query: "Instruct: ...\nQuery: "
    """

    def __init__(
        self,
        model_id: str,
        trust_remote_code: bool = False,
        passage_prefix: str = "",
        query_prefix: str = "",
    ):
        self._model_id = model_id
        self._trust_remote_code = trust_remote_code
        self._passage_prefix = passage_prefix
        self._query_prefix = query_prefix

    def _get_model(self):
        if self._model_id not in _hf_model_cache:
            from sentence_transformers import SentenceTransformer
            print(f"[vector_store] Loading HuggingFace model: {self._model_id}")
            _hf_model_cache[self._model_id] = SentenceTransformer(
                self._model_id,
                trust_remote_code=self._trust_remote_code,
            )
            print(f"[vector_store] Model loaded: {self._model_id}")
        return _hf_model_cache[self._model_id]

    def _encode(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        embeddings = self._get_model().encode(
            input,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=16,
        )
        return embeddings.tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Document indexing — applies passage_prefix."""
        if self._passage_prefix:
            input = [self._passage_prefix + t for t in input]
        return self._encode(input)

    def encode_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Query search — applies query_prefix. Use this for retrieval."""
        if self._query_prefix:
            input = [self._query_prefix + t for t in input]
        return self._encode(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Called by ChromaDB 1.5+ for query_texts= path."""
        return self.encode_query(input)

    # ── ChromaDB 1.5+ EmbeddingFunction Protocol ─────────────────────────────

    @staticmethod
    def name() -> str:
        return "huggingface_sentence_transformers"

    @staticmethod
    def build_from_config(config: dict) -> "_HuggingFaceEmbeddingFn":
        return _HuggingFaceEmbeddingFn(
            model_id=config["model_id"],
            trust_remote_code=config.get("trust_remote_code", False),
            passage_prefix=config.get("passage_prefix", ""),
            query_prefix=config.get("query_prefix", ""),
        )

    def get_config(self) -> dict:
        return {
            "model_id": self._model_id,
            "trust_remote_code": self._trust_remote_code,
            "passage_prefix": self._passage_prefix,
            "query_prefix": self._query_prefix,
        }


# ── Supported embedding models ────────────────────────────────────────────────

EMBEDDING_MODELS: dict[str, dict] = {
    "local:all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2 (로컬 ONNX)",
        "provider": "local",
        "dimensions": 384,
        "requires_key": None,
        "description": "API 키 불필요 · 첫 실행 시 ~25MB 자동 다운로드",
    },
    "openai:text-embedding-3-small": {
        "name": "text-embedding-3-small (OpenAI)",
        "provider": "openai",
        "dimensions": 1536,
        "requires_key": "openai",
        "description": "빠르고 저렴 · OpenAI API 키 필요",
    },
    "openai:text-embedding-3-large": {
        "name": "text-embedding-3-large (OpenAI)",
        "provider": "openai",
        "dimensions": 3072,
        "requires_key": "openai",
        "description": "고품질 · OpenAI API 키 필요",
    },
    # ── HuggingFace / sentence-transformers models ───────────────────────────
    "hf:BAAI/bge-m3": {
        "name": "bge-m3",
        "provider": "local",
        "dimensions": 1024,
        "requires_key": None,
        "description": "다국어 SOTA · 한국어 최강 · ~580MB",
        "hf_model_id": "BAAI/bge-m3",
        "trust_remote_code": False,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
    },
    "hf:nomic-ai/nomic-embed-text-v1.5": {
        "name": "nomic-embed-text-v1.5",
        "provider": "local",
        "dimensions": 768,
        "requires_key": None,
        "description": "범용 텍스트 임베딩 · ~270MB",
        "hf_model_id": "nomic-ai/nomic-embed-text-v1.5",
        "trust_remote_code": True,
        "passage_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    },
    "hf:jhgan/ko-sroberta-multitask": {
        "name": "ko-sroberta-multitask",
        "provider": "local",
        "dimensions": 768,
        "requires_key": None,
        "description": "한국어 특화 SRoBERTa · ~320MB",
        "hf_model_id": "jhgan/ko-sroberta-multitask",
        "trust_remote_code": False,
        "passage_prefix": "",
        "query_prefix": "",
    },
    "hf:jinaai/jina-embeddings-v4": {
        "name": "jina-embeddings-v4",
        "provider": "local",
        "dimensions": 2048,
        "requires_key": None,
        "description": "다국어 멀티모달 · Matryoshka 지원 · ~2.4GB",
        "hf_model_id": "jinaai/jina-embeddings-v4",
        "trust_remote_code": True,
        "passage_prefix": "",
        "query_prefix": "",
    },
    "hf:Qwen/qwen3-embedding-4b": {
        "name": "qwen3-embedding-4b",
        "provider": "local",
        "dimensions": 2560,
        "requires_key": None,
        "description": "Qwen3 4B · 한국어/다국어 우수 · ~8GB RAM 필요",
        "hf_model_id": "Qwen/qwen3-embedding-4b",
        "trust_remote_code": False,
        "passage_prefix": "",
        "query_prefix": "Instruct: Given a user question, retrieve relevant document passages.\nQuery: ",
    },
    "hf:google/embeddinggemma-300m": {
        "name": "embeddinggemma-300m",
        "provider": "local",
        "dimensions": 1024,
        "requires_key": None,
        "description": "Google Gemma 기반 경량 임베딩 · ~300MB",
        "hf_model_id": "google/embeddinggemma-300m",
        "trust_remote_code": False,
        "passage_prefix": "",
        "query_prefix": "",
    },
    "hf:intfloat/multilingual-e5-small": {
        "name": "multilingual-e5-small",
        "provider": "local",
        "dimensions": 384,
        "requires_key": None,
        "description": "Microsoft 다국어 E5 · 한국어 포함 100개 언어 · ~220MB",
        "hf_model_id": "intfloat/multilingual-e5-small",
        "trust_remote_code": False,
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
    },
}

# ── ChromaDB client (singleton) ───────────────────────────────────────────────

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        os.makedirs(app_settings.chroma_data_dir, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=app_settings.chroma_data_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


# ── Embedding functions ───────────────────────────────────────────────────────

def _get_embedding_function(emb_id: str):
    """Return the appropriate ChromaDB EmbeddingFunction for the given model."""
    if emb_id == "local:all-MiniLM-L6-v2":
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            return DefaultEmbeddingFunction()
        except Exception:
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
            return ONNXMiniLM_L6_V2()

    if emb_id.startswith("openai:"):
        model_name = emb_id.split(":", 1)[1]
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        return OpenAIEmbeddingFunction(
            api_key=app_settings.openai_api_key,
            model_name=model_name,
        )

    if emb_id.startswith("hf:"):
        meta = EMBEDDING_MODELS.get(emb_id, {})
        hf_model_id = meta.get("hf_model_id", emb_id[3:])
        return _HuggingFaceEmbeddingFn(
            hf_model_id,
            trust_remote_code=meta.get("trust_remote_code", False),
            passage_prefix=meta.get("passage_prefix", ""),
            query_prefix=meta.get("query_prefix", ""),
        )

    raise ValueError(f"Unknown embedding model: {emb_id!r}")


# ── Collection helpers ────────────────────────────────────────────────────────

def _col_name(emb_id: str) -> str:
    """Stable, ChromaDB-safe collection name derived from embedding model ID."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", emb_id)
    return f"docs__{safe}"


def _get_collection(emb_id: str):
    ef = _get_embedding_function(emb_id)
    return _get_client().get_or_create_collection(
        name=_col_name(emb_id),
        embedding_function=ef,
        metadata={"hnsw:space": "cosine", "embedding_model": emb_id},
    )


# ── Public API ────────────────────────────────────────────────────────────────

_UPSERT_BATCH = 50  # max chunks per ChromaDB upsert call (avoids huge Ollama payloads)


def add_chunks_parent_child(
    doc_id: str,
    filename: str,
    children: list[str],
    parents: list[str],
    child_to_parent: list[int],
    emb_id: str,
) -> int:
    """Index child chunks with parent text stored in metadata.

    ChromaDB indexes children for precise retrieval.
    At query time, the parent text is returned as the LLM context.
    """
    col = _get_collection(emb_id)
    for start in range(0, len(children), _UPSERT_BATCH):
        batch = children[start:start + _UPSERT_BATCH]
        ids = [f"{doc_id}__c{start + i}" for i in range(len(batch))]
        metas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": start + i,
                "parent_text": parents[child_to_parent[start + i]],
                "parent_index": child_to_parent[start + i],
                "is_child": True,
            }
            for i in range(len(batch))
        ]
        col.upsert(documents=batch, metadatas=metas, ids=ids)
    return len(children)


def add_chunks(doc_id: str, filename: str, chunks: list[str], emb_id: str) -> int:
    """Embed and store chunks using the specified embedding model.

    Chunks are sent to the embedding backend in batches of _UPSERT_BATCH to
    prevent oversized payloads to Ollama (which processes input sequentially
    and can time out on large batches with heavy models like qwen3-embedding).
    """
    col = _get_collection(emb_id)
    for start in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[start:start + _UPSERT_BATCH]
        ids   = [f"{doc_id}__{start + i}" for i in range(len(batch))]
        metas = [{"doc_id": doc_id, "filename": filename, "chunk_index": start + i} for i in range(len(batch))]
        col.upsert(documents=batch, metadatas=metas, ids=ids)
    return len(chunks)


def remove_doc(doc_id: str, emb_id: str | None = None) -> None:
    """Remove all chunks for a document from one or all embedding collections."""
    targets = [emb_id] if emb_id else list(EMBEDDING_MODELS.keys())
    for eid in targets:
        try:
            col = _get_collection(eid)
            hits = col.get(where={"doc_id": doc_id})
            if hits["ids"]:
                col.delete(ids=hits["ids"])
        except Exception:
            pass


def batch_query(
    texts: list[str],
    emb_id: str,
    n_results: int = 5,
    similarity_threshold: float = 0.0,
) -> list[list[tuple[str, float]]]:
    """Embed all texts in ONE batch call, then run ChromaDB lookups per embedding.

    This is dramatically faster than calling query() N times for Ollama models,
    because Ollama processes a batch of inputs in a single /api/embed request
    instead of N sequential requests.
    """
    col = _get_collection(emb_id)
    total = col.count()
    if total == 0:
        return [[] for _ in texts]

    ef = _get_embedding_function(emb_id)
    # Use query prefix if supported, else fall back to __call__
    encode_fn = getattr(ef, "encode_query", ef)
    embeddings = encode_fn(texts)

    n = min(n_results, total)
    results_per_text: list[list[tuple[str, float]]] = []
    for emb_vec in embeddings:
        try:
            res = col.query(query_embeddings=[emb_vec], n_results=n,
                            include=["documents", "distances"])
            docs  = res["documents"][0] if res["documents"] else []
            dists = res["distances"][0]  if res.get("distances") else [0.0] * len(docs)
            scored = [(doc, max(0.0, 1.0 - dist)) for doc, dist in zip(docs, dists)]
            if similarity_threshold > 0.0:
                scored = [(d, s) for d, s in scored if s >= similarity_threshold]
            results_per_text.append(scored)
        except Exception as exc:
            print(f"[vector_store] batch_query error ({emb_id}): {exc}")
            results_per_text.append([])
    return results_per_text


def query(
    text: str,
    emb_id: str,
    n_results: int = 3,
    similarity_threshold: float = 0.0,
) -> list[tuple[str, float]]:
    """Return top-n chunks with cosine similarity scores (0–1, higher = more relevant).

    ChromaDB stores cosine *distance* (= 1 − cosine_similarity) so we convert:
        similarity = 1 − distance

    Chunks whose similarity is below `similarity_threshold` are dropped so that
    low-quality matches don't pollute the LLM context.
    """
    col = _get_collection(emb_id)
    total = col.count()
    if total == 0:
        return []
    n = min(n_results, total)
    try:
        results = col.query(query_texts=[text], n_results=n, include=["documents", "distances"])
        docs = results["documents"][0] if results["documents"] else []
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        scored = [(doc, max(0.0, 1.0 - dist)) for doc, dist in zip(docs, dists)]
        if similarity_threshold > 0.0:
            scored = [(doc, sim) for doc, sim in scored if sim >= similarity_threshold]
        return scored
    except Exception as exc:
        print(f"[vector_store] query error ({emb_id}): {exc}")
        return []


def query_with_meta(
    text: str,
    emb_id: str,
    n_results: int = 3,
    similarity_threshold: float = 0.0,
) -> list[tuple[str, float, dict]]:
    """Like query() but also returns ChromaDB metadata per chunk.

    Returns list of (context_text, score, metadata).
    For parent-child chunks, context_text is the parent text.
    metadata contains: doc_id, chunk_index, parent_text (optional), is_child (optional).
    """
    col = _get_collection(emb_id)
    total = col.count()
    if total == 0:
        return []
    n = min(n_results, total)
    try:
        ef = _get_embedding_function(emb_id)
        encode_fn = getattr(ef, "encode_query", ef)
        query_emb = encode_fn([text])[0]
        results = col.query(
            query_embeddings=[query_emb],
            n_results=n,
            include=["documents", "distances", "metadatas"],
        )
        docs = results["documents"][0] if results["documents"] else []
        dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
        metas = results["metadatas"][0] if results.get("metadatas") else [{} for _ in docs]

        output: list[tuple[str, float, dict]] = []
        for doc, dist, meta in zip(docs, dists, metas):
            score = max(0.0, 1.0 - dist)
            if similarity_threshold > 0.0 and score < similarity_threshold:
                continue
            # For parent-child: use parent_text as context
            context_text = meta.get("parent_text", doc) if meta.get("is_child") else doc
            output.append((context_text, score, meta or {}))
        return output
    except Exception as exc:
        print(f"[vector_store] query_with_meta error ({emb_id}): {exc}")
        return []


def semantic_query_as_chunks(
    text: str,
    emb_id: str,
    n_results: int = 5,
    similarity_threshold: float = 0.0,
) -> "list[RetrievedChunk]":
    """Semantic search returning RetrievedChunk objects."""
    from app.schemas import RetrievedChunk
    results = query_with_meta(text, emb_id, n_results, similarity_threshold)
    chunks = []
    for context_text, score, meta in results:
        raw_doc = meta.get("parent_text", context_text) if not meta.get("is_child") else context_text
        child_text = None
        if meta.get("is_child"):
            # context_text is already the parent; we need the original child
            # We can't easily recover it here without extra storage, so skip
            child_text = None
        chunks.append(RetrievedChunk(
            content=context_text,
            matched_text=child_text,
            chunk_index=meta.get("chunk_index", 0),
            doc_id=meta.get("doc_id", ""),
            semantic_score=score,
            bm25_score=None,
            final_score=score,
        ))
    return chunks


def hybrid_query_rrf(
    text: str,
    emb_id: str,
    n_results: int = 5,
    similarity_threshold: float = 0.0,
    rrf_k: int = 60,
    doc_ids: "list[str] | None" = None,
) -> "list[RetrievedChunk]":
    """Hybrid semantic + BM25 search with Reciprocal Rank Fusion.

    rrf_score(d) = Σ 1/(k + rank_in_list(d))
    Documents appearing high in multiple ranked lists get the highest scores.
    """
    from app.schemas import RetrievedChunk
    from app.services import bm25_store

    # Semantic results (more candidates for RRF)
    sem_results = query_with_meta(text, emb_id, n_results * 2)
    # BM25 results
    bm25_results = bm25_store.query_all(text, n_results * 2, doc_ids)

    # Accumulate RRF scores keyed by (doc_id, chunk_index)
    rrf: dict[tuple, float] = {}
    chunk_data: dict[tuple, dict] = {}

    for rank, (context_text, sem_score, meta) in enumerate(sem_results):
        key = (meta.get("doc_id", ""), meta.get("chunk_index", rank))
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        chunk_data[key] = {
            "content": context_text,
            "matched_text": None,
            "chunk_index": meta.get("chunk_index", rank),
            "doc_id": meta.get("doc_id", ""),
            "sem_score": sem_score,
            "bm25_score": 0.0,
        }

    for rank, (bm25_content, bm25_score, chunk_idx, doc_id) in enumerate(bm25_results):
        key = (doc_id, chunk_idx)
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        if key in chunk_data:
            chunk_data[key]["bm25_score"] = bm25_score
        else:
            chunk_data[key] = {
                "content": bm25_content,
                "matched_text": None,
                "chunk_index": chunk_idx,
                "doc_id": doc_id,
                "sem_score": 0.0,
                "bm25_score": bm25_score,
            }

    # Sort by RRF score and take top n
    sorted_keys = sorted(rrf.keys(), key=lambda k: rrf[k], reverse=True)[:n_results]

    result: list[RetrievedChunk] = []
    for key in sorted_keys:
        d = chunk_data[key]
        sem = d["sem_score"]
        if similarity_threshold > 0.0 and sem < similarity_threshold and d["bm25_score"] == 0:
            continue
        result.append(RetrievedChunk(
            content=d["content"],
            matched_text=d.get("matched_text"),
            chunk_index=d["chunk_index"],
            doc_id=d["doc_id"],
            semantic_score=sem if sem > 0 else None,
            bm25_score=d["bm25_score"] if d["bm25_score"] > 0 else None,
            final_score=rrf[key],
        ))
    return result


def bm25_only_query(
    text: str,
    n_results: int = 5,
    doc_ids: "list[str] | None" = None,
) -> "list[RetrievedChunk]":
    """BM25-only retrieval returning RetrievedChunk objects."""
    from app.schemas import RetrievedChunk
    from app.services import bm25_store

    raw = bm25_store.query_all(text, n_results, doc_ids)
    result = []
    for content, bm25_score, chunk_idx, doc_id in raw:
        result.append(RetrievedChunk(
            content=content,
            matched_text=None,
            chunk_index=chunk_idx,
            doc_id=doc_id,
            semantic_score=None,
            bm25_score=bm25_score,
            final_score=bm25_score,
        ))
    return result


# Alias kept for compatibility with debug router
def query(
    text: str,
    emb_id: str,
    n_results: int = 3,
    similarity_threshold: float = 0.0,
) -> list[tuple[str, float]]:
    """Semantic search. Returns (context_text, score) pairs."""
    return [(ctx, score) for ctx, score, _ in query_with_meta(text, emb_id, n_results, similarity_threshold)]


def collection_count(emb_id: str) -> int:
    try:
        return _get_collection(emb_id).count()
    except Exception:
        return 0


def get_sample_chunks(
    emb_id: str,
    doc_id: str | None = None,
    max_chunks: int = 60,
) -> list[dict]:
    """컬렉션에서 샘플 청크를 가져옵니다 (테스트 케이스 자동 생성용).

    doc_id가 주어지면 해당 문서의 청크만, None이면 전체에서 샘플링.
    반환값: [{"text": str, "doc_id": str, "filename": str}, ...]
    """
    try:
        col = _get_collection(emb_id)
        kwargs: dict = {"include": ["documents", "metadatas"]}
        if doc_id:
            kwargs["where"] = {"doc_id": doc_id}
        result = col.get(**kwargs)

        docs  = result.get("documents") or []
        metas = result.get("metadatas") or []

        # 본문이 없거나 너무 짧은 청크 제외
        items = [
            {"text": d, "doc_id": m.get("doc_id", ""), "filename": m.get("filename", "")}
            for d, m in zip(docs, metas)
            if d and len(d.strip()) >= 80
        ]

        # 고르게 샘플링
        if len(items) <= max_chunks:
            return items
        step = len(items) / max_chunks
        return [items[int(i * step)] for i in range(max_chunks)]
    except Exception as exc:
        print(f"[vector_store] get_sample_chunks error: {exc}")
        return []


# ── Model catalogue ───────────────────────────────────────────────────────────

async def get_embedding_model_infos() -> list[EmbeddingModelInfo]:
    """Return all embedding model infos.

    HuggingFace models are always available (auto-downloaded on first use).
    OpenAI models require an API key.
    """
    models = []
    for eid, meta in EMBEDDING_MODELS.items():
        if meta["requires_key"] == "openai":
            available = bool(app_settings.openai_api_key)
        else:
            available = True

        models.append(EmbeddingModelInfo(
            id=eid,
            name=meta["name"],
            provider=meta["provider"],
            available=available,
            dimensions=meta["dimensions"],
            description=meta["description"],
        ))
    return models
