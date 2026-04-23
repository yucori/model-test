"""
Debug / diagnostic endpoints — not part of the benchmarking pipeline.

POST /api/debug/retrieve        →  single-query retrieval across embedding models
POST /api/debug/batch-retrieve  →  multi-query batch retrieval for auto-evaluation
"""
import asyncio
import math
import statistics

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import vector_store

router = APIRouter(prefix="/api/debug", tags=["debug"])

# Threshold used for MRR / Hit@K / distribution calculations.
# Queries with top-1 score below this are counted as "not found".
_RELEVANCE_THRESHOLD = 0.5


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


def _compute_aggregates(
    queries: list[str],
    hits_per_query: list[list[tuple[str, float]]],
    top_k: int,
) -> dict:
    """Compute rich retrieval quality metrics for one embedding model.

    Metrics
    -------
    avg_score       : mean cosine similarity of ALL returned chunks
    avg_top1_score  : mean cosine similarity of rank-1 chunk per query
    score_std       : std-dev of rank-1 scores (consistency; lower = more stable)
    hit_rate        : fraction of queries that returned ≥1 chunk
    hit_at_1        : fraction of queries where rank-1 score ≥ _RELEVANCE_THRESHOLD
    hit_at_3        : fraction of queries where ≥1 of top-3 chunks ≥ threshold
    mrr             : Mean Reciprocal Rank — mean(1/rank) of first chunk ≥ threshold
                      (rank counts from 1; 0 if no chunk passes threshold)
    score_distribution : counts of queries whose rank-1 score falls in:
                      high  ≥ 0.70  ("잘 찾음")
                      mid   0.50–0.70 ("보통")
                      low   < 0.50   ("검색 실패")
    rank_scores     : [avg_score_at_rank1, …, avg_score_at_rankK] — shows drop-off
    avg_retrieved   : average number of chunks returned per query
    relevance_threshold : threshold value used for MRR / Hit calculations
    """
    n = len(queries)
    if n == 0:
        return {
            "avg_score": 0.0, "avg_top1_score": 0.0, "score_std": 0.0,
            "hit_rate": 0.0, "hit_at_1": 0.0, "hit_at_3": 0.0, "mrr": 0.0,
            "score_distribution": {"high": 0, "mid": 0, "low": 0},
            "rank_scores": [], "avg_retrieved": 0.0,
            "relevance_threshold": _RELEVANCE_THRESHOLD,
            "total_queries": 0, "hit_queries": 0,
        }

    all_scores: list[float] = []
    top1_scores: list[float] = []
    hit_count = 0
    hit_at_1_count = 0
    hit_at_3_count = 0
    mrr_sum = 0.0
    dist_high = dist_mid = dist_low = 0
    retrieved_counts: list[int] = []

    # Accumulate scores per rank position across all queries
    rank_buckets: list[list[float]] = [[] for _ in range(top_k)]

    for hits in hits_per_query:
        if hits:
            hit_count += 1

        top1 = hits[0][1] if hits else 0.0
        top1_scores.append(top1)
        retrieved_counts.append(len(hits))

        for s in (s for _, s in hits):
            all_scores.append(s)

        for rank_idx, (_, s) in enumerate(hits):
            if rank_idx < top_k:
                rank_buckets[rank_idx].append(s)

        # Hit@1
        if hits and hits[0][1] >= _RELEVANCE_THRESHOLD:
            hit_at_1_count += 1

        # Hit@3
        if any(s >= _RELEVANCE_THRESHOLD for _, s in hits[:3]):
            hit_at_3_count += 1

        # MRR — reciprocal rank of the first chunk ≥ threshold
        for rank, (_, s) in enumerate(hits, start=1):
            if s >= _RELEVANCE_THRESHOLD:
                mrr_sum += 1.0 / rank
                break

        # Score distribution (based on rank-1 score)
        if top1 >= 0.70:
            dist_high += 1
        elif top1 >= 0.50:
            dist_mid += 1
        else:
            dist_low += 1

    # Aggregate score stats
    avg_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    avg_top1 = round(sum(top1_scores) / n, 4)

    # Population std-dev of top-1 scores
    if n > 1:
        mean = sum(top1_scores) / n
        variance = sum((s - mean) ** 2 for s in top1_scores) / n
        score_std = round(math.sqrt(variance), 4)
    else:
        score_std = 0.0

    rank_scores = [
        round(sum(bucket) / len(bucket), 4) if bucket else 0.0
        for bucket in rank_buckets
    ]

    # ── 동적 임계값 (백분위 기반) ────────────────────────────────────────────
    # 각 모델 고유의 점수 분포 중앙값을 임계값으로 사용.
    # 고정 임계값 0.5는 점수가 압축된 모델에 유리해지는 문제를 보완한다.
    # 중앙값을 쓰면 "이 모델에서 상위 50%에 해당하는가"라는 공통 기준이 된다.
    dynamic_thr = round(statistics.median(top1_scores), 4) if n >= 2 else _RELEVANCE_THRESHOLD

    dyn_hit1 = dyn_hit3 = 0
    dyn_mrr_sum = 0.0
    for i, hits in enumerate(hits_per_query):
        s1 = top1_scores[i]
        if s1 >= dynamic_thr:
            dyn_hit1 += 1
        if any(s >= dynamic_thr for _, s in hits[:3]):
            dyn_hit3 += 1
        for rank, (_, s) in enumerate(hits, start=1):
            if s >= dynamic_thr:
                dyn_mrr_sum += 1.0 / rank
                break

    return {
        "avg_score":           avg_score,
        "avg_top1_score":      avg_top1,
        "score_std":           score_std,
        "hit_rate":            round(hit_count      / n, 4),
        "hit_at_1":            round(hit_at_1_count / n, 4),
        "hit_at_3":            round(hit_at_3_count / n, 4),
        "mrr":                 round(mrr_sum        / n, 4),
        "score_distribution":  {"high": dist_high, "mid": dist_mid, "low": dist_low},
        "rank_scores":         rank_scores,
        "avg_retrieved":       round(sum(retrieved_counts) / n, 2),
        "relevance_threshold": _RELEVANCE_THRESHOLD,
        "total_queries":       n,
        "hit_queries":         hit_count,
        # 동적 임계값 지표 (모델별 점수 분포 중앙값 기준)
        "dynamic_threshold":   dynamic_thr,
        "dynamic_hit_at_1":    round(dyn_hit1    / n, 4),
        "dynamic_hit_at_3":    round(dyn_hit3    / n, 4),
        "dynamic_mrr":         round(dyn_mrr_sum / n, 4),
    }


@router.post("/batch-retrieve")
async def debug_batch_retrieve(body: BatchRetrieveRequest):
    """Run multiple queries against multiple embedding models.

    Optimization: per model, all queries are embedded in ONE batch call to the
    backend (Ollama / OpenAI / local). Previously N_queries × N_models individual
    embedding calls were made; now it is N_models calls total. For Ollama this
    reduces wall-clock time from minutes to seconds on 20+ questions.
    """
    queries = [q for q in body.queries if q.strip()]
    if not queries:
        raise HTTPException(400, "queries must not be empty")
    if not body.embedding_model_ids:
        raise HTTPException(400, "embedding_model_ids must not be empty")

    async def _batch_one_model(emb_id: str) -> tuple[str, list[list[tuple[str, float]]]]:
        hits_per_query = await asyncio.to_thread(
            vector_store.batch_query,
            queries,
            emb_id,
            body.top_k,
            body.similarity_threshold,
        )
        return emb_id, hits_per_query

    # Run each model's batch in parallel (models run concurrently, queries batched within each)
    pairs = await asyncio.gather(*[_batch_one_model(eid) for eid in body.embedding_model_ids])

    # Organise: results[query][emb_id] = list[chunk]
    results: dict[str, dict[str, list]] = {q: {} for q in queries}
    aggregates: dict[str, dict] = {}

    for emb_id, hits_per_query in pairs:
        for query, hits in zip(queries, hits_per_query):
            results[query][emb_id] = [{"text": t, "score": round(s, 4)} for t, s in hits]

        aggregates[emb_id] = _compute_aggregates(queries, hits_per_query, body.top_k)

    return {"top_k": body.top_k, "results": results, "aggregates": aggregates}
