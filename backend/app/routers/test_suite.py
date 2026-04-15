"""
Test suite router — CRUD for test questions.
"""
import uuid

from fastapi import APIRouter, HTTPException

from app import state
from app.schemas import CreateTestQuestion, TestQuestion, UpdateTestQuestion

router = APIRouter(prefix="/api/test-suite", tags=["test-suite"])


@router.get("", response_model=list[TestQuestion])
async def list_questions(category: str | None = None):
    questions = list(state.test_questions.values())
    if category:
        questions = [q for q in questions if q.category == category]
    return questions


@router.get("/categories")
async def list_categories():
    cats = sorted({q.category for q in state.test_questions.values()})
    return {"categories": cats}


@router.post("", response_model=TestQuestion)
async def create_question(body: CreateTestQuestion):
    q_id = f"q_custom_{uuid.uuid4().hex[:8]}"
    question = TestQuestion(id=q_id, **body.model_dump())
    state.test_questions[q_id] = question
    return question


@router.put("/{q_id}", response_model=TestQuestion)
async def update_question(q_id: str, body: UpdateTestQuestion):
    q = state.test_questions.get(q_id)
    if not q:
        raise HTTPException(404, "Question not found")

    data = q.model_dump()
    for field, val in body.model_dump(exclude_none=True).items():
        data[field] = val
    state.test_questions[q_id] = TestQuestion(**data)
    return state.test_questions[q_id]


@router.delete("/{q_id}")
async def delete_question(q_id: str):
    if q_id not in state.test_questions:
        raise HTTPException(404, "Question not found")
    del state.test_questions[q_id]
    return {"deleted": q_id}
