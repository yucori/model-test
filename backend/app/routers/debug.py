"""
Debug / diagnostic endpoints — not part of the benchmarking pipeline.

POST /api/debug/retrieve        →  single-query retrieval across embedding models
POST /api/debug/batch-retrieve  →  multi-query batch retrieval for auto-evaluation
"""
import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import vector_store

router = APIRouter(prefix="/api/debug", tags=["debug"])


class RetrieveRequest(BaseModel):
    query: str
    embedding_model_ids: list[str]
    top_k: int = 5
    similarity_threshold: float = 0.0


class BatchRetrieveRequest(BaseModel):
    queries: list[str]
    embedding_model_ids: list[str]
    top_k: int = 5
    similarity_threshold: float = 0.0


@router.post("/retrieve")
async def debug_retrieve(body: RetrieveRequest):
    """Return top-k chunks per embedding model for a given query (no LLM call)."""
    if not body.query.strip():
        raise HTTPException(400, "query must not be empty")
    if not body.embedding_model_ids:
        raise HTTPException(400, "embedding_model_ids must not be empty")

    async def _query_one(emb_id: str):
        hits = await asyncio.to_thread(
            vector_store.query,
            body.query,
            emb_id,
            body.top_k,
            body.similarity_threshold,
        )
        return emb_id, [{"text": text, "score": round(score, 4)} for text, score in hits]

    tasks = [_query_one(eid) for eid in body.embedding_model_ids]
    pairs = await asyncio.gather(*tasks)

    return {
        "query": body.query,
        "top_k": body.top_k,
        "results": {emb_id: chunks for emb_id, chunks in pairs},
    }


@router.post("/batch-retrieve")
async def debug_batch_retrieve(body: BatchRetrieveRequest):
    """Run multiple queries against multiple embedding models in parallel.

    Returns per-query, per-model retrieval results plus aggregate stats.
    Used for automatic embedding evaluation with the full test question set.
    """
    queries = [q for q in body.queries if q.strip()]
    if not queries:
        raise HTTPException(400, "queries must not be empty")
    if not body.embedding_model_ids:
        raise HTTPException(400, "embedding_model_ids must not be empty")

    async def _query_one(query: str, emb_id: str):
        hits = await asyncio.to_thread(
            vector_store.query,
            query,
            emb_id,
            body.top_k,
            body.similarity_threshold,
        )
        return query, emb_id, [{"text": text, "score": round(score, 4)} for text, score in hits]

    tasks = [_query_one(q, eid) for q in queries for eid in body.embedding_model_ids]
    triples = await asyncio.gather(*tasks)

    # Organise: results[query][emb_id] = list[chunk]
    results: dict[str, dict[str, list]] = {}
    for query, emb_id, chunks in triples:
        if query not in results:
            results[query] = {}
        results[query][emb_id] = chunks

    # Per-model aggregates
    aggregates: dict[str, dict] = {}
    for emb_id in body.embedding_model_ids:
        scores = []
        top1_scores = []
        hit_count = 0  # queries that returned at least one chunk
        for query in queries:
            chunks = results.get(query, {}).get(emb_id, [])
            if chunks:
                hit_count += 1
                top1_scores.append(chunks[0]["score"])
                scores.extend(c["score"] for c in chunks)
        aggregates[emb_id] = {
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "avg_top1_score": round(sum(top1_scores) / len(top1_scores), 4) if top1_scores else 0.0,
            "hit_rate": round(hit_count / len(queries), 4) if queries else 0.0,
            "total_queries": len(queries),
            "hit_queries": hit_count,
        }

    return {
        "top_k": body.top_k,
        "results": results,
        "aggregates": aggregates,
    }
