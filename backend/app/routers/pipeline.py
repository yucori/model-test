"""
Pipeline stage comparison endpoints.

These endpoints support the step-by-step RAG pipeline analysis workflow:

  Stage 2: POST /api/pipeline/chunk-compare
    Given a document + parser + list of chunking variants,
    returns stats for each variant (chunk count, size distribution, samples).
    No ChromaDB involved — pure text analysis.

  Stage 3: POST /api/pipeline/search-compare
    Given an indexed document + embedding model + list of search strategies,
    runs test queries against each strategy and returns the retrieved chunks.
    Requires the document to be vectorized first (BM25 index also built).
"""
import asyncio
import os
import statistics

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import state
from app.config import settings
from app.schemas import (
    ChunkCompareResult,
    ChunkVariantConfig,
    ChunkVariantStats,
    SearchCompareResult,
    SearchQueryResult,
    SearchChunkResult,
    EmbTestCase,
    EmbTestResult,
    EmbModelResult,
    EmbCompareResult,
)
from app.services import document_processor, vector_store, bm25_store

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


# ── Stage 2: Chunk comparison ─────────────────────────────────────────────────

class ChunkCompareRequest(BaseModel):
    doc_id: str
    parser: str = "pdfplumber"
    variants: list[ChunkVariantConfig]


@router.post("/chunk-compare", response_model=ChunkCompareResult)
async def chunk_compare(body: ChunkCompareRequest):
    """Compare multiple chunking strategies on a document.

    Returns statistics and sample chunks for each variant.
    No embedding or ChromaDB — pure text analysis.
    """
    doc = state.documents.get(body.doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    file_path = os.path.join(settings.upload_dir, f"{body.doc_id}.{doc.file_type}")
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found on disk")

    # Extract text once
    try:
        text = await asyncio.to_thread(
            document_processor.extract_text, file_path, body.parser
        )
    except Exception as exc:
        raise HTTPException(500, f"텍스트 추출 실패: {exc}")

    variant_stats: list[ChunkVariantStats] = []

    for variant in body.variants:
        try:
            if variant.strategy == "parent_child":
                pc = await asyncio.to_thread(
                    document_processor.chunk_text_parent_child,
                    text, variant.chunk_size,
                )
                chunks = pc["children"]
            else:
                chunks = await asyncio.to_thread(
                    document_processor.chunk_text,
                    text,
                    variant.chunk_size,
                    variant.overlap,
                    variant.strategy,
                )

            if not chunks:
                variant_stats.append(ChunkVariantStats(
                    config=variant,
                    chunk_count=0,
                    avg_size=0,
                    median_size=0,
                    min_size=0,
                    max_size=0,
                    size_buckets={"< 200": 0, "200–400": 0, "400–600": 0, "> 600": 0},
                    sample_chunks=[],
                ))
                continue

            import re as _re
            sizes = [len(c) for c in chunks]
            buckets = {
                "< 200": sum(1 for s in sizes if s < 200),
                "200–400": sum(1 for s in sizes if 200 <= s < 400),
                "400–600": sum(1 for s in sizes if 400 <= s < 600),
                "> 600": sum(1 for s in sizes if s >= 600),
            }
            structure_aligned = sum(
                1 for c in chunks if _re.match(r'^#{2,3} ', c.lstrip())
            )

            variant_stats.append(ChunkVariantStats(
                config=variant,
                chunk_count=len(chunks),
                avg_size=round(sum(sizes) / len(sizes), 1),
                median_size=round(statistics.median(sizes), 1),
                min_size=min(sizes),
                max_size=max(sizes),
                size_buckets=buckets,
                structure_aligned_count=structure_aligned,
                sample_chunks=chunks[:5],
            ))
        except Exception as exc:
            print(f"[pipeline] chunk-compare error for variant {variant.label}: {exc}")
            variant_stats.append(ChunkVariantStats(
                config=variant,
                chunk_count=0,
                avg_size=0,
                median_size=0,
                min_size=0,
                max_size=0,
                size_buckets={"< 200": 0, "200–400": 0, "400–600": 0, "> 600": 0},
                structure_aligned_count=0,
                sample_chunks=[f"오류: {exc}"],
            ))

    return ChunkCompareResult(
        doc_id=body.doc_id,
        filename=doc.filename,
        parser=body.parser,
        variants=variant_stats,
    )


# ── Stage 3: Search strategy comparison ──────────────────────────────────────

class SearchCompareRequest(BaseModel):
    doc_id: str
    emb_model_id: str
    queries: list[str]
    strategies: list[str]   # "semantic" | "bm25" | "hybrid_rrf"
    top_k: int = 5
    similarity_threshold: float = 0.0


@router.post("/search-compare", response_model=SearchCompareResult)
async def search_compare(body: SearchCompareRequest):
    """Compare search strategies on an already-indexed document.

    For each (strategy, query) pair, returns the top-K retrieved chunks
    with their scores. Useful for visually inspecting retrieval quality.

    Prerequisites: document must be vectorized (for semantic/hybrid)
    and BM25 index must be built (happens automatically during vectorize).
    """
    doc = state.documents.get(body.doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    if not body.queries:
        raise HTTPException(400, "최소 1개의 쿼리가 필요합니다")

    strategy_labels = {
        "semantic": "Semantic (벡터 유사도)",
        "bm25": "BM25 (키워드)",
        "hybrid_rrf": "Hybrid RRF (BM25 + Semantic)",
    }

    query_results: list[SearchQueryResult] = []

    for query in body.queries:
        for strategy in body.strategies:
            label = strategy_labels.get(strategy, strategy)
            try:
                if strategy == "bm25":
                    raw = await asyncio.to_thread(
                        vector_store.bm25_only_query,
                        query,
                        body.top_k,
                        [body.doc_id],
                    )
                elif strategy == "hybrid_rrf":
                    raw = await asyncio.to_thread(
                        vector_store.hybrid_query_rrf,
                        query,
                        body.emb_model_id,
                        body.top_k,
                        body.similarity_threshold,
                        60,
                        [body.doc_id],
                    )
                else:  # semantic
                    raw = await asyncio.to_thread(
                        vector_store.semantic_query_as_chunks,
                        query,
                        body.emb_model_id,
                        body.top_k,
                        body.similarity_threshold,
                    )

                chunks = [
                    SearchChunkResult(
                        content=c.content,
                        matched_text=c.matched_text,
                        chunk_index=c.chunk_index,
                        doc_id=c.doc_id,
                        semantic_score=c.semantic_score,
                        bm25_score=c.bm25_score,
                        final_score=c.final_score,
                    )
                    for c in raw
                ]
                avg_score = sum(c.final_score for c in chunks) / len(chunks) if chunks else 0.0

                query_results.append(SearchQueryResult(
                    strategy=strategy,
                    label=label,
                    query=query,
                    chunks=chunks,
                    avg_score=round(avg_score, 4),
                ))
            except Exception as exc:
                print(f"[pipeline] search-compare error ({strategy}, {query!r}): {exc}")
                query_results.append(SearchQueryResult(
                    strategy=strategy,
                    label=label,
                    query=query,
                    chunks=[],
                    avg_score=0.0,
                ))

    return SearchCompareResult(
        doc_id=body.doc_id,
        emb_model_id=body.emb_model_id,
        query_results=query_results,
        strategies_tested=body.strategies,
        queries_tested=body.queries,
    )


# ── Preview chunks (wizard용) ────────────────────────────────────────────────

def _suggest_phrase(text: str) -> str:
    """청크 텍스트에서 expected_contains 후보 구문을 추출.

    헤딩/목차 줄을 건너뛰고 첫 번째 본문 줄의 앞 40자를 반환.
    본문 줄이 없으면 전체 텍스트의 앞 40자.
    """
    for line in text.split('\n'):
        ls = line.strip()
        if not ls:
            continue
        if ls.startswith('#'):
            continue
        if ls.count('|') >= 2:
            continue
        return ls[:40]
    return text.strip()[:40]


@router.get("/preview-chunks")
async def preview_chunks(
    query: str,
    emb_model_id: str,
    top_k: int = 5,
):
    """쿼리에 대해 실제 검색되는 청크를 미리 봅니다.

    테스트 케이스 wizard에서 사용: 정답 청크를 확인하고
    expected_contains 후보 구문을 제안합니다.
    """
    if not query.strip():
        raise HTTPException(400, "query가 비어있습니다")

    has_any = any(
        emb_model_id in doc.processed_embeddings
        for doc in state.documents.values()
    )
    if not has_any:
        raise HTTPException(400, f"{emb_model_id} 모델로 벡터화된 문서가 없습니다")

    try:
        raw = await asyncio.to_thread(
            vector_store.semantic_query_as_chunks,
            query,
            emb_model_id,
            top_k,
            0.0,
        )
    except Exception as exc:
        raise HTTPException(500, f"검색 오류: {exc}")

    return {
        "query": query,
        "emb_model_id": emb_model_id,
        "chunks": [
            {
                "rank": i + 1,
                "content": c.content,
                "matched_text": c.matched_text,
                "chunk_index": c.chunk_index,
                "doc_id": c.doc_id,
                "final_score": c.final_score,
                "suggested_phrase": _suggest_phrase(c.matched_text or c.content),
            }
            for i, c in enumerate(raw)
        ],
    }


# ── Hit detection helper ─────────────────────────────────────────────────────

def _is_content_hit(text: str, needle: str) -> bool:
    """키워드가 목차/제목 줄이 아닌 본문 줄에 존재하는지 확인.

    다음 줄은 히트에서 제외:
      - 마크다운 헤딩 줄 (## / ###로 시작)
      - 파이프 구분 목차 줄 (| 2개 이상 — "제1장 x | 제2장 y" 형식)
    """
    lower_needle = needle.lower()
    for line in text.split('\n'):
        ls = line.strip()
        if not ls:
            continue
        if lower_needle not in ls.lower():
            continue
        # 마크다운 헤딩
        if ls.startswith('#'):
            continue
        # 파이프 구분 목차 (제N장 a | 제M장 b)
        if ls.count('|') >= 2:
            continue
        # 본문 줄에서 발견
        return True
    return False


# ── Stage 4: Embedding model comparison ──────────────────────────────────────

class EmbCompareRequest(BaseModel):
    test_cases: list[EmbTestCase]
    emb_model_ids: list[str]
    top_k: int = 5


@router.post("/emb-compare", response_model=EmbCompareResult)
async def emb_compare(body: EmbCompareRequest):
    """Evaluate retrieval hit-rate for each embedding model.

    For each model × test_case pair, retrieves top_k chunks and checks
    whether `expected_contains` appears in any of the retrieved chunk texts.
    Returns Hit@1, Hit@3, Hit@K per model.
    """
    if not body.test_cases:
        raise HTTPException(400, "테스트 케이스가 없습니다")
    if not body.emb_model_ids:
        raise HTTPException(400, "임베딩 모델을 1개 이상 선택하세요")

    model_results: list[EmbModelResult] = []

    for emb_id in body.emb_model_ids:
        # Check if any document has been vectorized with this model
        has_any = any(
            emb_id in doc.processed_embeddings
            for doc in state.documents.values()
        )
        if not has_any:
            model_results.append(EmbModelResult(
                emb_model_id=emb_id,
                test_results=[],
                hit_at_1=0.0,
                hit_at_3=0.0,
                hit_at_k=0.0,
                available=False,
            ))
            continue

        test_results: list[EmbTestResult] = []

        for tc in body.test_cases:
            try:
                raw = await asyncio.to_thread(
                    vector_store.semantic_query_as_chunks,
                    tc.query,
                    emb_id,
                    body.top_k,
                    0.0,
                )
                chunks = [
                    SearchChunkResult(
                        content=c.content,
                        matched_text=c.matched_text,
                        chunk_index=c.chunk_index,
                        doc_id=c.doc_id,
                        semantic_score=c.semantic_score,
                        bm25_score=c.bm25_score,
                        final_score=c.final_score,
                    )
                    for c in raw
                ]

                # Find hit rank (1-indexed), None if not found in top-k
                hit_rank: int | None = None
                needle = tc.expected_contains.lower()
                for i, chunk in enumerate(chunks, start=1):
                    text = chunk.matched_text or chunk.content
                    if needle in text.lower() and _is_content_hit(text, needle):
                        hit_rank = i
                        break

                test_results.append(EmbTestResult(
                    test_case_id=tc.id,
                    query=tc.query,
                    expected_contains=tc.expected_contains,
                    hit_rank=hit_rank,
                    chunks=chunks,
                ))
            except Exception as exc:
                print(f"[pipeline] emb-compare error ({emb_id}, {tc.id}): {exc}")
                test_results.append(EmbTestResult(
                    test_case_id=tc.id,
                    query=tc.query,
                    expected_contains=tc.expected_contains,
                    hit_rank=None,
                    chunks=[],
                ))

        n = len(test_results)
        hit_at_1 = sum(1 for r in test_results if r.hit_rank == 1) / n if n else 0.0
        hit_at_3 = sum(1 for r in test_results if r.hit_rank is not None and r.hit_rank <= 3) / n if n else 0.0
        hit_at_k = sum(1 for r in test_results if r.hit_rank is not None) / n if n else 0.0

        model_results.append(EmbModelResult(
            emb_model_id=emb_id,
            test_results=test_results,
            hit_at_1=round(hit_at_1, 4),
            hit_at_3=round(hit_at_3, 4),
            hit_at_k=round(hit_at_k, 4),
            available=True,
        ))

    return EmbCompareResult(
        model_results=model_results,
        top_k=body.top_k,
        total_cases=len(body.test_cases),
    )


# ── Test case auto-generation ─────────────────────────────────────────────────

_GEN_PROMPT = """다음은 농산물 직판매 쇼핑몰 정책 문서의 한 단락입니다.

---
{chunk}
---

이 단락을 읽고 아래 JSON을 출력하세요.

조건:
- question: 고객이 이 내용에 대해 물어볼 수 있는 자연스러운 한국어 질문 (1문장)
- expected_contains: 이 단락 본문에 그대로 포함된 고유 문구 (10~35자). 반드시 위 텍스트에 존재해야 함. ## 헤딩이나 목차 형식 불가.
- category: 배송 / 결제 / 환불 / 회원 / 고객서비스 / 반품교환 / 품질보증 중 하나

JSON만 출력 (마크다운 없이):
{{"question": "...", "expected_contains": "...", "category": "..."}}"""


async def _generate_one_case(chunk_text: str, judge_model: str) -> dict | None:
    """청크 하나로 테스트 케이스를 생성합니다."""
    import json, re
    from app.config import settings

    prompt = _GEN_PROMPT.format(chunk=chunk_text.strip()[:600])

    try:
        if judge_model.startswith("gemini"):
            from google import genai
            client = genai.Client(api_key=settings.gemini_api_key)
            r = await client.aio.models.generate_content(
                model=judge_model,
                contents=prompt,
            )
            raw = r.text or ""
        elif judge_model.startswith("openrouter:"):
            from openai import AsyncOpenAI
            actual_model = judge_model.removeprefix("openrouter:")
            client = AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )
            r = await client.chat.completions.create(
                model=actual_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            raw = r.choices[0].message.content or ""
        else:
            import httpx
            async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=60) as c:
                resp = await c.post("/api/chat", json={
                    "model": judge_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                })
                resp.raise_for_status()
                raw = resp.json().get("message", {}).get("content", "")

        # JSON 추출
        raw = raw.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            raw = m.group()
        data = json.loads(raw)

        q  = str(data.get("question", "")).strip()
        ec = str(data.get("expected_contains", "")).strip()
        cat = str(data.get("category", "")).strip()

        if not q or not ec:
            return None
        # expected_contains가 실제 청크에 존재하는지 검증
        if ec.lower() not in chunk_text.lower():
            return None

        return {"question": q, "expected_contains": ec, "category": cat}
    except Exception as exc:
        print(f"[pipeline] _generate_one_case error: {exc}")
        return None


class GenerateTestCasesRequest(BaseModel):
    emb_model_id: str
    doc_id: str | None = None     # None = 전체 문서에서 샘플링
    num_cases: int = 20
    judge_model: str = "gemini-2.0-flash"


@router.post("/generate-test-cases")
async def generate_test_cases(body: GenerateTestCasesRequest):
    """청크에서 LLM으로 테스트 케이스를 자동 생성합니다."""
    import asyncio as _aio

    has_any = any(
        body.emb_model_id in doc.processed_embeddings
        for doc in state.documents.values()
    )
    if not has_any:
        raise HTTPException(400, f"{body.emb_model_id} 모델로 벡터화된 문서가 없습니다")

    # 샘플 청크 가져오기
    sample_chunks = await _aio.to_thread(
        vector_store.get_sample_chunks,
        body.emb_model_id,
        body.doc_id,
        body.num_cases * 2,  # 실패 여분 확보
    )

    if not sample_chunks:
        raise HTTPException(400, "청크가 없습니다. 먼저 문서를 벡터화하세요.")

    # 병렬 생성 (최대 num_cases * 2개 시도)
    tasks = [
        _generate_one_case(c["text"], body.judge_model)
        for c in sample_chunks[:body.num_cases * 2]
    ]
    results = await _aio.gather(*tasks)

    # None 제거 + 중복 question 제거 + num_cases 제한
    seen_q: set[str] = set()
    cases = []
    for i, r in enumerate(results):
        if r is None:
            continue
        if r["question"] in seen_q:
            continue
        seen_q.add(r["question"])
        cases.append({
            "id": f"gen-{i}-{int(__import__('time').time())}",
            **r,
        })
        if len(cases) >= body.num_cases:
            break

    return {"cases": cases, "total": len(cases)}


# ── BM25 status ───────────────────────────────────────────────────────────────

@router.get("/bm25-status")
async def bm25_status():
    """Return which documents have a BM25 index built."""
    return {
        "indexed_docs": bm25_store.indexed_docs(),
        "total": len(bm25_store.indexed_docs()),
    }
