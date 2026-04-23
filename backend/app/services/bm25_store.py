"""
In-memory BM25 index for keyword-based retrieval.

Indexes are built per-document when chunks are added (via add_chunks).
The BM25 index is rebuilt from scratch whenever a document is re-indexed,
so re-vectorizing a document with new chunking settings automatically
updates the BM25 index to match.

Usage:
    # After vectorizing a document:
    bm25_store.add_chunks(doc_id, chunks)

    # At query time:
    results = bm25_store.query(text, doc_id, n=5)
    all_results = bm25_store.query_all(text, n=10)
"""
import re
from typing import Optional

# In-memory index: doc_id → (chunk_texts, BM25Okapi)
_indexes: dict[str, tuple[list[str], object]] = {}


def _tokenize(text: str) -> list[str]:
    """Tokenize Korean + English text for BM25.

    Splits on whitespace/punctuation, keeps Korean eojeol and English words.
    Lowercases everything for case-insensitive matching.
    """
    return re.findall(r"[가-힣]+|[a-zA-Z0-9]+", text.lower())


def add_chunks(doc_id: str, chunks: list[str]) -> None:
    """Build (or rebuild) the BM25 index for a document."""
    if not chunks:
        return
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [_tokenize(c) for c in chunks]
        # BM25Okapi requires at least one non-empty tokenized doc
        if not any(tokenized):
            return
        bm25 = BM25Okapi(tokenized)
        _indexes[doc_id] = (list(chunks), bm25)
        print(f"[bm25_store] Indexed {len(chunks)} chunks for doc {doc_id}")
    except Exception as exc:
        print(f"[bm25_store] Failed to index {doc_id}: {exc}")


def query(
    text: str,
    doc_id: str,
    n: int = 5,
) -> list[tuple[str, float, int]]:
    """Query the BM25 index for a specific document.

    Returns list of (chunk_text, bm25_score, chunk_index).
    Chunks with score == 0 are excluded.
    """
    if doc_id not in _indexes:
        return []
    chunks, bm25 = _indexes[doc_id]
    tokens = _tokenize(text)
    if not tokens:
        return []
    try:
        scores = bm25.get_scores(tokens)
    except Exception:
        return []

    # Sort by score descending, take top n, exclude zeros
    indexed = [(chunks[i], float(scores[i]), i) for i in range(len(scores)) if scores[i] > 0]
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:n]


def query_all(
    text: str,
    n: int = 10,
    doc_ids: Optional[list[str]] = None,
) -> list[tuple[str, float, int, str]]:
    """Query all (or specified) indexed documents.

    Returns list of (chunk_text, bm25_score, chunk_index, doc_id).
    Results are merged and sorted by score descending.
    """
    targets = doc_ids if doc_ids is not None else list(_indexes.keys())
    all_results: list[tuple[str, float, int, str]] = []

    for did in targets:
        if did not in _indexes:
            continue
        chunks, bm25 = _indexes[did]
        tokens = _tokenize(text)
        if not tokens:
            continue
        try:
            scores = bm25.get_scores(tokens)
        except Exception:
            continue
        for i, score in enumerate(scores):
            if score > 0:
                all_results.append((chunks[i], float(score), i, did))

    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[:n]


def remove_doc(doc_id: str) -> None:
    """Remove the BM25 index for a document."""
    _indexes.pop(doc_id, None)


def has_index(doc_id: str) -> bool:
    """Return True if a BM25 index exists for this document."""
    return doc_id in _indexes


def indexed_docs() -> list[str]:
    """Return list of all indexed doc IDs."""
    return list(_indexes.keys())
