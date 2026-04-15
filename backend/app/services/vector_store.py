"""
Per-embedding-model ChromaDB collections.

Each embedding model gets its own isolated collection so that:
  - Vector dimensions don't clash (MiniLM=384, OpenAI-small=1536, etc.)
  - We can compare how different embeddings affect RAG retrieval quality.

The embedding function is set on the collection and called automatically by
ChromaDB when add(documents=...) or query(query_texts=...) is called.
"""
import os
import re
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings as app_settings
from app.schemas import EmbeddingModelInfo

# ── Custom Ollama embedding function (long timeout for large models) ──────────

class _OllamaEmbeddingFn:
    """ChromaDB-compatible embedding function that calls Ollama /api/embed.

    ChromaDB's built-in OllamaEmbeddingFunction has a short default timeout
    which causes failures when a large model (e.g. qwen3-embedding 4.7 GB)
    needs time to load on the first call. This wrapper uses a 300-second timeout.
    """

    def __init__(self, model_name: str, timeout: int = 300):
        self._model = model_name
        self._timeout = timeout
        self._base_url = app_settings.ollama_base_url

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        import httpx
        with httpx.Client(base_url=self._base_url, timeout=self._timeout) as client:
            r = client.post("/api/embed", json={"model": self._model, "input": input})
            if not r.is_success:
                try:
                    detail = r.json().get("error", r.text)
                except Exception:
                    detail = r.text
                raise RuntimeError(f"Ollama embed {r.status_code}: {detail}")
            return r.json()["embeddings"]

    # ── ChromaDB 1.5+ EmbeddingFunction Protocol ─────────────────────────────
    # All four methods below are required by chromadb.api.types.EmbeddingFunction.

    @staticmethod
    def name() -> str:
        return "ollama_embedding"

    @staticmethod
    def build_from_config(config: dict) -> "_OllamaEmbeddingFn":
        return _OllamaEmbeddingFn(model_name=config["model_name"], timeout=config.get("timeout", 300))

    def get_config(self) -> dict:
        return {"model_name": self._model, "timeout": self._timeout}

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """ChromaDB calls this for query embeddings; delegates to __call__."""
        return self.__call__(input)


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
    # ── Ollama embedding models ──────────────────────────────────────────────
    "ollama:nomic-embed-text:latest": {
        "name": "nomic-embed-text (Ollama)",
        "provider": "ollama",
        "dimensions": 768,
        "requires_key": None,
        "description": "범용 고성능 텍스트 임베딩 · Ollama 필요",
    },
    "ollama:qwen3-embedding:latest": {
        "name": "qwen3-embedding (Ollama)",
        "provider": "ollama",
        "dimensions": 2048,
        "requires_key": None,
        "description": "Qwen3 고차원 임베딩 · 한국어 우수 · Ollama 필요",
    },
    "ollama:embeddinggemma:latest": {
        "name": "embeddinggemma (Ollama)",
        "provider": "ollama",
        "dimensions": 768,
        "requires_key": None,
        "description": "Gemma 기반 임베딩 · Ollama 필요",
    },
    "ollama:bge-m3:latest": {
        "name": "bge-m3 (Ollama)",
        "provider": "ollama",
        "dimensions": 1024,
        "requires_key": None,
        "description": "다국어 SOTA · 한국어 최강 · 권장 (ollama pull bge-m3)",
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

    if emb_id.startswith("ollama:"):
        model_name = emb_id.split(":", 1)[1]
        return _OllamaEmbeddingFn(model_name)

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

def add_chunks(doc_id: str, filename: str, chunks: list[str], emb_id: str) -> int:
    """Embed and store chunks using the specified embedding model."""
    col = _get_collection(emb_id)
    ids = [f"{doc_id}__{i}" for i in range(len(chunks))]
    metas = [{"doc_id": doc_id, "filename": filename, "chunk_index": i} for i in range(len(chunks))]
    col.upsert(documents=chunks, metadatas=metas, ids=ids)
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


def collection_count(emb_id: str) -> int:
    try:
        return _get_collection(emb_id).count()
    except Exception:
        return 0


# ── Model catalogue ───────────────────────────────────────────────────────────

async def get_embedding_model_infos() -> list[EmbeddingModelInfo]:
    """Return all embedding model infos, with Ollama availability checked live."""
    # Fetch available Ollama model names once
    ollama_available: set[str] = set()
    try:
        import httpx
        async with httpx.AsyncClient(base_url=app_settings.ollama_base_url, timeout=5) as c:
            r = await c.get("/api/tags")
            ollama_available = {m["name"] for m in r.json().get("models", [])}
    except Exception:
        pass

    models = []
    for eid, meta in EMBEDDING_MODELS.items():
        if meta["requires_key"] == "openai":
            available = bool(app_settings.openai_api_key)
        elif meta["provider"] == "ollama":
            # model_name is the part after "ollama:"
            ollama_model = eid.split(":", 1)[1]
            available = ollama_model in ollama_available
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
