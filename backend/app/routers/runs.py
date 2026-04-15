"""
Runs router — create, stream (SSE), list, get results and comparison.
"""
import asyncio
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sse_starlette.sse import EventSourceResponse

from app import state
from app.schemas import (
    PairSummary,
    RunComparison,
    TestRun,
    TestRunConfig,
    TestRunStatus,
)
from app.services import test_runner, vector_store

router = APIRouter(prefix="/api/runs", tags=["runs"])

# Short name helper
def _short(model_id: str | None) -> str:
    if not model_id:
        return "—"
    return (
        model_id
        .replace("claude-", "")
        .replace("text-embedding-", "emb-")
        .replace("local:", "")
        .replace("openai:", "")
        .split(":")[0]
    )


@router.post("", response_model=TestRun)
async def create_run(config: TestRunConfig, background_tasks: BackgroundTasks):
    if not config.llm_model_ids:
        raise HTTPException(400, "최소 1개의 LLM 모델을 선택해주세요")
    if config.rag_enabled and not config.embedding_model_ids:
        raise HTTPException(400, "RAG 사용 시 최소 1개의 임베딩 모델을 선택해주세요")

    all_qs = list(state.test_questions.values())
    if config.question_ids is not None:
        questions = [q for q in all_qs if q.id in config.question_ids]
        if not questions:
            raise HTTPException(400, "선택한 질문 ID가 존재하지 않습니다")
    else:
        questions = all_qs
    if not questions:
        raise HTTPException(400, "테스트 질문이 없습니다. 먼저 질문을 추가해주세요.")

    # Warn if documents aren't indexed for requested embedding models
    if config.rag_enabled:
        for emb_id in config.embedding_model_ids:
            if vector_store.collection_count(emb_id) == 0:
                print(f"[runs] Warning: no documents indexed for {emb_id}")

    run_id = str(uuid.uuid4())
    if config.rag_enabled:
        n_pairs = len(config.embedding_model_ids) * len(config.llm_model_ids)
    else:
        n_pairs = len(config.llm_model_ids)

    name = config.run_name or f"Run {datetime.now(timezone.utc).strftime('%m-%d %H:%M')}"
    run = TestRun(
        id=run_id,
        name=name,
        config=config,
        status=TestRunStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        total_tests=n_pairs * len(questions),
    )
    state.runs[run_id] = run
    background_tasks.add_task(test_runner.start_run, run, questions)
    return run


@router.get("", response_model=list[TestRun])
async def list_runs():
    return sorted(state.runs.values(), key=lambda r: r.created_at, reverse=True)


@router.get("/{run_id}", response_model=TestRun)
async def get_run(run_id: str):
    r = state.runs.get(run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    return r


@router.get("/{run_id}/stream")
async def stream_run(run_id: str):
    if run_id not in state.runs:
        raise HTTPException(404, "Run not found")
    run = state.runs[run_id]

    async def generator():
        yield {"event": "ping", "data": "connected"}

        if run.status in (TestRunStatus.COMPLETED, TestRunStatus.FAILED):
            for result in state.results.get(run_id, []):
                yield {"event": "result", "data": json.dumps(result.model_dump(mode="json"))}
            yield {"event": "complete", "data": json.dumps({"status": run.status})}
            return

        for _ in range(30):
            if run_id in state.run_queues:
                break
            await asyncio.sleep(0.1)

        queue = state.run_queues.get(run_id)
        if queue is None:
            yield {"event": "complete", "data": json.dumps({"status": "error"})}
            return

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "heartbeat"}
                continue

            yield {"event": event["type"], "data": json.dumps(event.get("data", {}))}
            if event["type"] == "complete":
                break

    return EventSourceResponse(generator())


@router.get("/{run_id}/results")
async def get_results(run_id: str):
    if run_id not in state.runs:
        raise HTTPException(404, "Run not found")
    return state.results.get(run_id, [])


@router.get("/{run_id}/comparison", response_model=RunComparison)
async def get_comparison(run_id: str):
    run = state.runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    results = state.results.get(run_id, [])
    if not results:
        raise HTTPException(404, "No results yet")

    # Group by (embedding_model_id, llm_model_id)
    pair_data: dict[tuple, list] = defaultdict(list)
    for r in results:
        pair_data[(r.embedding_model_id, r.llm_model_id)].append(r)

    def _avg_score(res_list: list, key: str) -> float:
        scored = [r for r in res_list if r.scores]
        vals = [getattr(r.scores, key) for r in scored]
        return sum(vals) / len(vals) if vals else 0.0

    def _avg_retrieval(res_list: list) -> float:
        """Average cosine similarity across all retrieved chunks in all results."""
        all_scores = [s for r in res_list for s in r.retrieval_scores]
        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    pair_summaries: list[PairSummary] = []
    for (emb_id, llm_id), res_list in pair_data.items():
        label = f"{_short(emb_id)} + {_short(llm_id)}" if emb_id else _short(llm_id)
        pair_summaries.append(PairSummary(
            embedding_model_id=emb_id,
            llm_model_id=llm_id,
            pair_label=label,
            avg_latency_ms=sum(r.latency_ms for r in res_list) / len(res_list),
            avg_relevance=_avg_score(res_list, "relevance"),
            avg_accuracy=_avg_score(res_list, "accuracy"),
            avg_helpfulness=_avg_score(res_list, "helpfulness"),
            avg_korean_fluency=_avg_score(res_list, "korean_fluency"),
            avg_overall=_avg_score(res_list, "overall"),
            total_tests=len(res_list),
            failed_tests=sum(1 for r in res_list if r.error),
            avg_completion_tokens=sum(r.completion_tokens for r in res_list) / len(res_list),
            avg_retrieval_score=_avg_retrieval(res_list),
        ))

    pair_summaries.sort(key=lambda p: p.avg_overall, reverse=True)

    # Aggregate by LLM (across all embeddings)
    llm_scores: dict[str, list[float]] = defaultdict(list)
    for ps in pair_summaries:
        llm_scores[ps.llm_model_id].append(ps.avg_overall)
    llm_avg = {k: sum(v) / len(v) for k, v in llm_scores.items()}

    # Aggregate by embedding (across all LLMs)
    emb_scores: dict[str, list[float]] = defaultdict(list)
    for ps in pair_summaries:
        if ps.embedding_model_id:
            emb_scores[ps.embedding_model_id].append(ps.avg_overall)
    emb_avg = {k: sum(v) / len(v) for k, v in emb_scores.items()}

    # By category breakdown
    q_map = {q.id: q for q in state.test_questions.values()}
    cat_map: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r.scores:
            continue
        cat = q_map.get(r.question_id)
        label = f"{_short(r.embedding_model_id)} + {_short(r.llm_model_id)}" if r.embedding_model_id else _short(r.llm_model_id)
        cat_map[cat.category if cat else "기타"][label].append(r.scores.overall)

    by_category = {
        cat: {label: sum(scores) / len(scores) for label, scores in model_data.items()}
        for cat, model_data in cat_map.items()
    }

    return RunComparison(
        run_id=run_id,
        run_name=run.name,
        pair_summaries=pair_summaries,
        llm_avg=llm_avg,
        emb_avg=emb_avg,
        by_category=by_category,
    )


@router.delete("/{run_id}")
async def delete_run(run_id: str):
    if run_id not in state.runs:
        raise HTTPException(404, "Run not found")
    del state.runs[run_id]
    state.results.pop(run_id, None)
    state.run_queues.pop(run_id, None)
    return {"deleted": run_id}
