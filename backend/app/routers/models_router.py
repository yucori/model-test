"""
Models router — endpoints for parsers, embedding models, and LLM models.
"""
from fastapi import APIRouter
from app.schemas import EmbeddingModelInfo, LLMModelInfo, ParserInfo
from app.services.llm_clients import get_available_llm_models
from app.services.vector_store import get_embedding_model_infos

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/parsers", response_model=list[ParserInfo])
async def list_parsers():
    """Document parsers — OCR/text extraction backends."""
    from app.services.document_processor import get_parser_infos
    return get_parser_infos()


@router.get("/chunk-strategies")
async def list_chunk_strategies():
    """Chunking strategy catalogue with metadata."""
    from app.services.document_processor import get_chunk_strategy_infos
    return get_chunk_strategy_infos()


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
