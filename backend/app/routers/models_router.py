"""
Models router — separate endpoints for embedding models and LLM models.
"""
from fastapi import APIRouter
from app.schemas import EmbeddingModelInfo, LLMModelInfo
from app.services.llm_clients import get_available_llm_models
from app.services.vector_store import get_embedding_model_infos

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/embedding", response_model=list[EmbeddingModelInfo])
async def list_embedding_models():
    """
    Embedding models — used to index documents and retrieve relevant context.
    Different embedding models can retrieve different chunks for the same query,
    which affects the quality of the LLM's answer downstream.
    """
    return await get_embedding_model_infos()


@router.get("/llm", response_model=list[LLMModelInfo])
async def list_llm_models():
    """
    LLM (generation) models — given the retrieved context, these produce the
    customer service answer. Testing multiple LLMs with the same context lets
    you isolate generation quality from retrieval quality.
    """
    return await get_available_llm_models()
