"""
Test runner — executes the full (embedding × LLM) × question matrix.

Test matrix when RAG is enabled:
  for question in questions:
    for emb_id in embedding_model_ids:        ← varies RAG retrieval quality
      context = vector_store.query(q, emb_id)
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
from app.schemas import TestResult, TestRun, TestRunConfig, TestRunStatus, TestQuestion
from app.services import llm_clients, evaluator, vector_store

_CS_SYSTEM = """당신은 농산물 직판매 쇼핑몰의 친절하고 전문적인 고객 서비스 상담원입니다.
고객의 질문에 아래 참고 자료를 활용하여 명확하고 도움이 되는 답변을 제공하세요.

[참고 자료]
{context}

참고 자료에 없는 내용에 대해서는 "정확한 안내를 위해 고객 센터(☎ 1588-0000)로 문의해 주시면 친절히 안내해 드리겠습니다." 라고 안내하세요.
반드시 한국어로 답변하세요."""


async def start_run(run: TestRun, questions: list[TestQuestion]) -> None:
    run_id = run.id
    queue: asyncio.Queue = asyncio.Queue()
    state.run_queues[run_id] = queue
    state.results[run_id] = []

    cfg: TestRunConfig = run.config

    # Build the test pairs
    if cfg.rag_enabled and cfg.embedding_model_ids:
        # Full matrix: each embedding × each LLM
        pairs: list[tuple[str | None, str]] = [
            (emb_id, llm_id)
            for emb_id in cfg.embedding_model_ids
            for llm_id in cfg.llm_model_ids
        ]
    else:
        # RAG disabled: just test LLMs with empty context
        pairs = [(None, llm_id) for llm_id in cfg.llm_model_ids]

    total = len(pairs) * len(questions)
    state.runs[run_id].total_tests = total
    state.runs[run_id].status = TestRunStatus.RUNNING

    completed = 0

    try:
        for question in questions:
            for (emb_id, llm_id) in pairs:
                # ── 1. Retrieve context ───────────────────────────────────────
                context: list[str] = []
                retrieval_scores: list[float] = []
                if emb_id is not None:
                    # Run synchronous ChromaDB call in a thread so we don't
                    # block the event loop during embedding API calls.
                    scored = await asyncio.to_thread(
                        vector_store.query,
                        question.question,
                        emb_id,
                        cfg.top_k,
                        cfg.similarity_threshold,
                    )
                    context = [doc for doc, _ in scored]
                    retrieval_scores = [score for _, score in scored]

                context_text = "\n---\n".join(context) if context else "참고 자료 없음"
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
                        context=context,
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
                    retrieved_context=context,
                    retrieval_scores=retrieval_scores,
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
