"""
Test runner — executes the full (embedding × LLM) × question matrix.

Test matrix when RAG is enabled:
  for question in questions:
    for emb_id in embedding_model_ids:        ← varies RAG retrieval quality
      context = retrieve(q, emb_id, strategy) ← semantic / BM25 / hybrid
      for llm_id in llm_model_ids:            ← varies answer generation quality
        response = call_llm(llm_id, context, q)
        score = judge(response)

When RAG is disabled:
  for question in questions:
    for llm_id in llm_model_ids:
      response = call_llm(llm_id, context=[], q)
      score = judge(response)
"""
import asyncio
import uuid
from datetime import datetime, timezone

from app import state
from app.schemas import (
    RetrievedChunk, SearchStrategy,
    TestResult, TestRun, TestRunConfig, TestRunStatus, TestQuestion,
)
from app.services import llm_clients, evaluator, vector_store

_CS_SYSTEM = """당신은 농산물 직판매 쇼핑몰의 친절하고 전문적인 고객 서비스 상담원입니다.
고객의 질문에 아래 참고 자료를 활용하여 명확하고 도움이 되는 답변을 제공하세요.

[참고 자료]
{context}

답변 시 반드시 다음 규칙을 따르세요:
1. 참고 자료에서 근거를 찾은 경우, 어떤 문서의 몇 조 몇 항(또는 해당 섹션명)을 근거로 했는지 명시하세요.
   예: "○○ 약관 제3조 제2항에 따르면..." 또는 "배송 정책 3.반품 기준에 의하면..."
2. 참고 자료에 조·항 구조가 없는 경우에도 출처 문서명이나 섹션명을 언급하세요.
3. 참고 자료에 없는 내용에 대해서는 "정확한 안내를 위해 고객 센터(☎ 1588-0000)로 문의해 주시면 친절히 안내해 드리겠습니다." 라고 안내하세요.
반드시 한국어로 답변하세요."""


async def _retrieve(
    question: str,
    emb_id: str,
    cfg: TestRunConfig,
) -> list[RetrievedChunk]:
    """Retrieve context chunks using the configured search strategy."""
    strategy = cfg.search_strategy

    if strategy == SearchStrategy.BM25:
        return await asyncio.to_thread(
            vector_store.bm25_only_query,
            question,
            cfg.top_k,
        )
    elif strategy == SearchStrategy.HYBRID_RRF:
        return await asyncio.to_thread(
            vector_store.hybrid_query_rrf,
            question,
            emb_id,
            cfg.top_k,
            cfg.similarity_threshold,
        )
    else:
        # SEMANTIC (default)
        return await asyncio.to_thread(
            vector_store.semantic_query_as_chunks,
            question,
            emb_id,
            cfg.top_k,
            cfg.similarity_threshold,
        )


async def start_run(run: TestRun, questions: list[TestQuestion]) -> None:
    run_id = run.id
    queue: asyncio.Queue = asyncio.Queue()
    state.run_queues[run_id] = queue
    state.results[run_id] = []

    cfg: TestRunConfig = run.config

    # Build the test pairs
    if cfg.rag_enabled and cfg.embedding_model_ids:
        pairs: list[tuple[str | None, str]] = [
            (emb_id, llm_id)
            for emb_id in cfg.embedding_model_ids
            for llm_id in cfg.llm_model_ids
        ]
    else:
        pairs = [(None, llm_id) for llm_id in cfg.llm_model_ids]

    total = len(pairs) * len(questions)
    state.runs[run_id].total_tests = total
    state.runs[run_id].status = TestRunStatus.RUNNING

    completed = 0

    try:
        for question in questions:
            for (emb_id, llm_id) in pairs:
                # ── 1. Retrieve context ───────────────────────────────────────
                retrieved_chunks: list[RetrievedChunk] = []
                if emb_id is not None:
                    retrieved_chunks = await _retrieve(question, emb_id, cfg)

                # Build plain text context for LLM + evaluator
                context_texts = [c.content for c in retrieved_chunks]
                context_text = "\n---\n".join(context_texts) if context_texts else "참고 자료 없음"
                system_prompt = _CS_SYSTEM.format(context=context_text)

                # ── 2. Generate answer ────────────────────────────────────────
                llm_resp = None
                response_text = ""
                llm_error: str | None = None

                try:
                    llm_resp = await llm_clients.call_model(
                        model_id=llm_id,
                        system=system_prompt,
                        question=question.question,
                    )
                    response_text = llm_resp.content
                except Exception as exc:
                    llm_error = str(exc)
                    print(f"[runner] LLM error ({llm_id}): {exc}")

                # ── 3. Evaluate ───────────────────────────────────────────────
                scores = None
                if response_text and not llm_error:
                    scores = await evaluator.evaluate_response(
                        question=question.question,
                        context=context_texts,
                        response=response_text,
                        judge_model=cfg.judge_model,
                    )

                # ── 4. Record result ──────────────────────────────────────────
                result = TestResult(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    embedding_model_id=emb_id,
                    llm_model_id=llm_id,
                    question_id=question.id,
                    question=question.question,
                    retrieved_chunks=retrieved_chunks,
                    response=response_text,
                    latency_ms=llm_resp.latency_ms if llm_resp else 0.0,
                    prompt_tokens=llm_resp.prompt_tokens if llm_resp else 0,
                    completion_tokens=llm_resp.completion_tokens if llm_resp else 0,
                    scores=scores,
                    error=llm_error,
                    completed_at=datetime.now(timezone.utc),
                )

                state.results[run_id].append(result)
                completed += 1
                state.runs[run_id].completed_tests = completed
                await queue.put({"type": "result", "data": result.model_dump(mode="json")})

        state.runs[run_id].status = TestRunStatus.COMPLETED
        state.runs[run_id].completed_at = datetime.now(timezone.utc)

    except Exception as exc:
        state.runs[run_id].status = TestRunStatus.FAILED
        state.runs[run_id].error = str(exc)
        print(f"[runner] Run {run_id} failed: {exc}")

    finally:
        await queue.put({"type": "complete"})
