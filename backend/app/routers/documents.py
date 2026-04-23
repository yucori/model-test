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
from app.services import document_processor, vector_store, bm25_store

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
    parser_id: str = "pdfplumber"     # legacy single-parser field (kept for compatibility)
    pdf_parser_id: str | None = None  # PDF-specific parser (overrides parser_id for PDFs)
    docx_parser_id: str = "python-docx"  # DOCX-specific parser
    chunk_size: int = 600            # characters per chunk
    overlap: int = 100               # overlap between consecutive chunks
    chunk_strategy: str = "paragraph"  # paragraph | sentence | fixed | semantic


@router.post("/vectorize")
async def vectorize(body: VectorizeRequest, background_tasks: BackgroundTasks):
    """Start background vectorization for the specified docs × embedding models."""
    if not body.embedding_model_ids:
        raise HTTPException(400, "임베딩 모델을 하나 이상 선택해주세요")

    # pdf_parser_id가 명시되면 사용, 없으면 legacy parser_id 사용
    effective_pdf_parser = body.pdf_parser_id or body.parser_id

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
                effective_pdf_parser, body.docx_parser_id,
                body.chunk_size, body.overlap, body.chunk_strategy,
            )

    return {"queued": [d.id for d in targets]}


async def _process_doc(
    doc_id: str,
    emb_ids: list[str],
    pdf_parser_id: str = "pdfplumber",
    docx_parser_id: str = "python-docx",
    chunk_size: int = 600,
    overlap: int = 100,
    chunk_strategy: str = "paragraph",
) -> None:
    doc = state.documents.get(doc_id)
    if not doc:
        return

    # 파일 확장자에 따라 파서 선택
    parser_id = pdf_parser_id if doc.file_type == "pdf" else docx_parser_id

    file_path = os.path.join(settings.upload_dir, f"{doc_id}.{doc.file_type}")
    try:
        text = await asyncio.to_thread(document_processor.extract_text, file_path, parser_id)

        # For parent-child, we need the full PC structure for ChromaDB
        # but for BM25 we index the parent chunks (richer context)
        if chunk_strategy == "parent_child":
            pc = await asyncio.to_thread(
                document_processor.chunk_text_parent_child,
                text, chunk_size,
            )
            children = pc["children"]
            parents = pc["parents"]
            child_to_parent = pc["child_to_parent"]
            # BM25 indexes parent chunks (search on richer text)
            bm25_chunks = parents
            chunks_for_count = children
        else:
            chunks = await asyncio.to_thread(
                document_processor.chunk_text,
                text, chunk_size, overlap, chunk_strategy,
            )
            children = None
            parents = None
            child_to_parent = None
            bm25_chunks = chunks
            chunks_for_count = chunks

        if not chunks_for_count:
            state.documents[doc_id].processing = False
            state.documents[doc_id].processing_error = "텍스트를 추출할 수 없습니다"
            return

        # Build BM25 index (shared across all embedding models for this doc)
        await asyncio.to_thread(bm25_store.add_chunks, doc_id, bm25_chunks)

    except Exception as exc:
        state.documents[doc_id].processing = False
        state.documents[doc_id].processing_error = str(exc)
        print(f"[documents] extraction failed {doc_id}: {exc}")
        return

    for emb_id in emb_ids:
        try:
            vector_store.remove_doc(doc_id, emb_id)
            if chunk_strategy == "parent_child":
                n = await asyncio.to_thread(
                    vector_store.add_chunks_parent_child,
                    doc_id, doc.filename, children, parents, child_to_parent, emb_id,
                )
            else:
                n = await asyncio.to_thread(
                    vector_store.add_chunks, doc_id, doc.filename, chunks, emb_id,
                )
            state.documents[doc_id].processed_embeddings[emb_id] = n
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


# ── Parser comparison (all parsers, one doc) ─────────────────────────────────

@router.get("/{doc_id}/compare-parsers")
async def compare_parsers(doc_id: str, chunk_size: int = 600, overlap: int = 100, chunk_strategy: str = "paragraph"):
    """Run all available parsers on a document and return metrics + text for each.

    Metrics per parser:
      - elapsed_ms: wall-clock extraction time
      - char_count, word_count, chunk_count
      - korean_word_count: Korean word (eojeol) count in cleaned text (markdown/symbols stripped)
      - info_density: ratio of meaningful chars (non-whitespace, non-symbol) to total
      - avg_chunk_size, min/max_chunk_size
      - text_preview (first 2000 chars)
      - error: non-null if the parser failed

    Scoring (for recommendation):
      40% normalized char_count + 40% normalized korean_word_count + 20% speed bonus

    korean_word_count is computed on symbol-stripped text so that Docling's
    Markdown structural characters (|, -, #, etc.) don't penalise it unfairly.
    """
    import re
    import time

    doc = state.documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    file_path = os.path.join(settings.upload_dir, f"{doc_id}.{doc.file_type}")
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found on disk")

    parsers = document_processor.get_parser_infos()
    file_ext = doc.file_type.lower()  # "pdf" or "docx"
    relevant = [p for p in parsers if file_ext in p.get("file_types", ["pdf", "docx"])]
    available = [p for p in relevant if p["available"]]
    unavailable = [p for p in relevant if not p["available"]]

    # Strip Markdown / structural symbols before measuring clean content.
    _MD_STRIP = re.compile(r"[|#*`~>\-]{1,}|^\s*[-=]{3,}\s*$", re.MULTILINE)

    def _clean(text: str) -> str:
        return _MD_STRIP.sub(" ", text)

    def _korean_word_count(text: str) -> int:
        """Korean eojeol count on symbol-stripped text (fair across all parsers)."""
        return len(re.findall(r"[\uAC00-\uD7A3]+", _clean(text)))

    def _clean_char_count(text: str) -> int:
        """Non-whitespace char count after stripping Markdown — fair across parsers."""
        return len(re.findall(r"\S", _clean(text)))

    def _structure_score(text: str) -> float:
        """Reward preserved document structure: tables and headings.

        Returns a value in [0, 1] representing structural richness.
        - Table rows  (lines containing ≥2 '|') count the most.
        - Heading lines (lines starting with '#') count less.
        Normalised so a document with many tables can approach 1.0.
        """
        lines = text.splitlines()
        table_rows = sum(1 for l in lines if l.count("|") >= 2)
        headings   = sum(1 for l in lines if re.match(r"^#{1,6}\s", l))
        # Cap at 200 table rows / 50 headings to avoid extreme outliers
        raw = min(table_rows, 200) / 200 * 0.8 + min(headings, 50) / 50 * 0.2
        return round(raw, 4)

    def _info_density(text: str) -> float:
        if not text:
            return 0.0
        meaningful = len(re.findall(r"[\w\uAC00-\uD7A3]", text))
        return round(meaningful / len(text), 4)

    async def _run_one(parser_id: str):
        t0 = time.perf_counter()
        try:
            text = await asyncio.to_thread(document_processor.extract_text, file_path, parser_id)
            elapsed = round((time.perf_counter() - t0) * 1000)
            chunks = document_processor.chunk_text(
                text, chunk_size=chunk_size, overlap=overlap, strategy=chunk_strategy
            )
            sizes = [len(c) for c in chunks]
            words = len(re.findall(r"\S+", text))
            return {
                "parser_id": parser_id,
                "elapsed_ms": elapsed,
                "char_count": len(text),
                "clean_char_count": _clean_char_count(text),
                "word_count": words,
                "chunk_count": len(chunks),
                "korean_word_count": _korean_word_count(text),
                "structure_score": _structure_score(text),
                "info_density": _info_density(text),
                "avg_chunk_size": round(sum(sizes) / len(sizes)) if sizes else 0,
                "min_chunk_size": min(sizes) if sizes else 0,
                "max_chunk_size": max(sizes) if sizes else 0,
                "text_preview": text[:2000],
                "error": None,
            }
        except Exception as exc:
            elapsed = round((time.perf_counter() - t0) * 1000)
            return {
                "parser_id": parser_id,
                "elapsed_ms": elapsed,
                "char_count": 0,
                "clean_char_count": 0,
                "word_count": 0,
                "chunk_count": 0,
                "korean_word_count": 0,
                "structure_score": 0.0,
                "info_density": 0.0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "text_preview": "",
                "error": str(exc),
            }

    results = await asyncio.gather(*[_run_one(p["id"]) for p in available])
    results = list(results)

    # Append unavailable parsers as stub rows so the UI can show install hints
    for p in unavailable:
        results.append({
            "parser_id": p["id"],
            "elapsed_ms": 0,
            "char_count": 0,
            "clean_char_count": 0,
            "word_count": 0,
            "chunk_count": 0,
            "korean_word_count": 0,
            "structure_score": 0.0,
            "info_density": 0.0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "text_preview": "",
            "error": p["description"],   # description contains install hint
            "unavailable": True,
        })

    # ── Recommendation ──────────────────────────────────────────────────────────
    # Scoring formula (weights sum to 1.0):
    #   35% clean_char_count  — content volume, Markdown-stripped so fair across parsers
    #   35% korean_word_count — primary quality signal, also Markdown-stripped
    #   20% structure_score   — table rows + headings preserved (rewards Docling)
    #   10% speed_bonus       — lower weight: Docling is slower but structurally richer
    successful = [r for r in results if r["error"] is None and r["clean_char_count"] > 0]

    recommendation = None
    if successful:
        max_clean  = max(r["clean_char_count"]  for r in successful) or 1
        max_korean = max(r["korean_word_count"] for r in successful) or 1
        max_struct = max(r["structure_score"]   for r in successful) or 1
        max_ms     = max(r["elapsed_ms"]        for r in successful) or 1
        for r in successful:
            r["score"] = round(
                0.35 * (r["clean_char_count"]  / max_clean)
                + 0.35 * (r["korean_word_count"] / max_korean)
                + 0.20 * (r["structure_score"]   / max_struct)
                + 0.10 * (1 - r["elapsed_ms"]    / max_ms),
                4,
            )
        best = max(successful, key=lambda r: r["score"])

        reasons = []
        if best["clean_char_count"] == max(r["clean_char_count"] for r in successful):
            reasons.append("가장 많은 텍스트 추출")
        if best["korean_word_count"] == max(r["korean_word_count"] for r in successful):
            reasons.append("한글 단어 수 최고")
        if best["structure_score"] == max(r["structure_score"] for r in successful) and best["structure_score"] > 0:
            reasons.append("표·제목 구조 보존 최고")
        if best["elapsed_ms"] == min(r["elapsed_ms"] for r in successful):
            reasons.append("처리 속도 최고")

        recommendation = {
            "parser_id": best["parser_id"],
            "score": best["score"],
            "reasons": reasons,
        }

    return {
        "doc_id": doc_id,
        "filename": doc.filename,
        "results": results,
        "recommendation": recommendation,
    }


# ── Text / chunk preview ─────────────────────────────────────────────────────

@router.get("/{doc_id}/text")
async def get_document_text(
    doc_id: str,
    parser: str = "pdfplumber",
    chunk_size: int = 600,
    overlap: int = 100,
    chunk_strategy: str = "paragraph",
):
    """Extract and return full text + chunk preview for a document (diagnostic)."""
    doc = state.documents.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    file_path = os.path.join(settings.upload_dir, f"{doc_id}.{doc.file_type}")
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found on disk")

    try:
        text = await asyncio.to_thread(document_processor.extract_text, file_path, parser)
        chunks = document_processor.chunk_text(
            text, chunk_size=chunk_size, overlap=overlap, strategy=chunk_strategy
        )
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
    bm25_store.remove_doc(doc_id)
    del state.documents[doc_id]
    return {"deleted": doc_id}
