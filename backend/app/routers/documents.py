"""
Documents router

Flow:
  POST /api/documents                      → save file only (no vectorization)
  POST /api/documents/vectorize            → vectorize all docs with given embedding models
  GET  /api/documents                      → list with processing status
  DELETE /api/documents/{id}
"""
import asyncio
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from app import state
from app.config import settings
from app.schemas import DocumentInfo
from app.services import document_processor, vector_store

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_TYPES = {".pdf", ".docx", ".doc"}


# ── Upload (file save only) ───────────────────────────────────────────────────

@router.post("", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """Save the uploaded file. Vectorization is triggered separately via /vectorize."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(400, f"지원하지 않는 파일 형식: {ext}. 허용: {ALLOWED_TYPES}")

    doc_id = str(uuid.uuid4())
    os.makedirs(settings.upload_dir, exist_ok=True)
    dest = os.path.join(settings.upload_dir, f"{doc_id}{ext}")

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = DocumentInfo(
        id=doc_id,
        filename=file.filename or f"document{ext}",
        file_type=ext.lstrip("."),
        size_bytes=os.path.getsize(dest),
        uploaded_at=datetime.now(timezone.utc),
    )
    state.documents[doc_id] = doc
    return doc


# ── Vectorize ─────────────────────────────────────────────────────────────────

class VectorizeRequest(BaseModel):
    embedding_model_ids: list[str]
    doc_ids: list[str] | None = None  # None = all documents
    chunk_size: int = 600   # characters per chunk
    overlap: int = 100      # overlap between consecutive chunks


@router.post("/vectorize")
async def vectorize(body: VectorizeRequest, background_tasks: BackgroundTasks):
    """Start background vectorization for the specified docs × embedding models."""
    if not body.embedding_model_ids:
        raise HTTPException(400, "임베딩 모델을 하나 이상 선택해주세요")

    targets = (
        [state.documents[did] for did in body.doc_ids if did in state.documents]
        if body.doc_ids is not None
        else list(state.documents.values())
    )
    if not targets:
        raise HTTPException(400, "처리할 문서가 없습니다")

    for doc in targets:
        if not doc.processing:
            state.documents[doc.id].processing = True
            state.documents[doc.id].processing_error = None
            background_tasks.add_task(
                _process_doc, doc.id, body.embedding_model_ids,
                body.chunk_size, body.overlap,
            )

    return {"queued": [d.id for d in targets]}


async def _process_doc(
    doc_id: str,
    emb_ids: list[str],
    chunk_size: int = 600,
    overlap: int = 100,
) -> None:
    doc = state.documents.get(doc_id)
    if not doc:
        return

    file_path = os.path.join(settings.upload_dir, f"{doc_id}.{doc.file_type}")
    try:
        text = document_processor.extract_text(file_path)
        chunks = document_processor.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            state.documents[doc_id].processing = False
            state.documents[doc_id].processing_error = "텍스트를 추출할 수 없습니다"
            return
    except Exception as exc:
        state.documents[doc_id].processing = False
        state.documents[doc_id].processing_error = str(exc)
        print(f"[documents] extraction failed {doc_id}: {exc}")
        return

    for emb_id in emb_ids:
        try:
            vector_store.remove_doc(doc_id, emb_id)
            n = await asyncio.to_thread(vector_store.add_chunks, doc_id, doc.filename, chunks, emb_id)
            state.documents[doc_id].processed_embeddings[emb_id] = n
            # Clear any previous error for this model on success
            state.documents[doc_id].embed_errors.pop(emb_id, None)
        except Exception as exc:
            err_msg = str(exc)
            print(f"[documents] embed failed {doc_id}/{emb_id}: {err_msg}")
            state.documents[doc_id].embed_errors[emb_id] = err_msg

    state.documents[doc_id].processing = False


# ── List ──────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[DocumentInfo])
async def list_documents():
    return list(state.documents.values())


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def vector_stats():
    emb_counts = {
        emb_id: await asyncio.to_thread(vector_store.collection_count, emb_id)
        for emb_id in vector_store.EMBEDDING_MODELS
    }
    return {"total_documents": len(state.documents), "embedding_chunk_counts": emb_counts}


# ── Text / chunk preview ─────────────────────────────────────────────────────

@router.get("/{doc_id}/text")
async def get_document_text(
    doc_id: str,
    chunk_size: int = 600,
    overlap: int = 100,
):
    """Extract and return full text + chunk preview for a document (diagnostic).

    chunk_size and overlap control the chunking strategy so the caller can
    preview different strategies without committing to vectorization.
    """
    doc = state.documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    file_path = os.path.join(settings.upload_dir, f"{doc_id}.{doc.file_type}")
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found on disk")

    try:
        text = await asyncio.to_thread(document_processor.extract_text, file_path)
        chunks = document_processor.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        sizes = [len(c) for c in chunks]
        return {
            "doc_id": doc_id,
            "filename": doc.filename,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "char_count": len(text),
            "chunk_count": len(chunks),
            "avg_chunk_size": round(sum(sizes) / len(sizes)) if sizes else 0,
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
            "text_preview": text[:3000],
            "chunks": [{"index": i, "text": c, "char_count": len(c)} for i, c in enumerate(chunks)],
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Delete ────────────────────────────────────────────────────────────────────

@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    doc = state.documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    file_path = os.path.join(settings.upload_dir, f"{doc_id}.{doc.file_type}")
    if os.path.exists(file_path):
        os.remove(file_path)

    await asyncio.to_thread(vector_store.remove_doc, doc_id)
    del state.documents[doc_id]
    return {"deleted": doc_id}
