"""
PDF / DOCX text extraction and chunking.

PDF extraction strategy:
  1. pdfplumber extracts the text layer (fast, accurate for digital PDFs).
  2. Pages with fewer than 50 characters of text are treated as scanned images.
  3. Those pages are rendered via pymupdf and fed to easyocr (Korean + English).
  4. The easyocr Reader is cached globally so the ~2 GB model loads only once.
"""
import re
from pathlib import Path
from typing import Optional

# ── OCR state ────────────────────────────────────────────────────────────────

_easyocr_reader = None  # lazy-initialized on first OCR call

_OCR_TEXT_THRESHOLD = 50  # chars — below this, page is treated as image-only


# ── Text extraction ─────────────────────────────────────────────────────────

def extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return _extract_docx(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(path: str) -> str:
    import pdfplumber

    pages_text: list[str] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if len(text) >= _OCR_TEXT_THRESHOLD:
                pages_text.append(text)
            else:
                # Text layer empty/sparse — try OCR
                ocr_text = _ocr_pdf_page(path, i)
                pages_text.append(ocr_text if ocr_text else text)

    return "\n\n".join(filter(None, pages_text))


def _ocr_pdf_page(pdf_path: str, page_index: int) -> Optional[str]:
    """Render one PDF page as an image and extract text with easyocr."""
    global _easyocr_reader
    try:
        import fitz  # pymupdf
        import easyocr
    except ImportError as e:
        print(f"[OCR] Package missing ({e}). Run: uv add pymupdf easyocr pillow")
        return None

    try:
        if _easyocr_reader is None:
            print("[OCR] Initializing easyocr — first run downloads ~2 GB models, please wait...")
            _easyocr_reader = easyocr.Reader(["ko", "en"], gpu=False)
            print("[OCR] easyocr ready.")

        doc = fitz.open(pdf_path)
        page = doc[page_index]
        # 2× scale for better OCR accuracy
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        results = _easyocr_reader.readtext(img_bytes, detail=0, paragraph=True)
        return "\n".join(results).strip() or None

    except Exception as e:
        print(f"[OCR] Failed on page {page_index} of {pdf_path}: {e}")
        return None


def _extract_docx(path: str) -> str:
    from docx import Document

    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    # Also grab table cells
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text.strip())
    return "\n\n".join(paragraphs)


# ── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[str]:
    """Split text into chunks, preferring paragraph boundaries over fixed character cuts.

    Strategy:
    1. Split the document on blank lines to get natural paragraphs.
    2. Greedily merge consecutive paragraphs until the chunk_size is reached.
    3. When a flush happens, carry the last paragraph as overlap into the next chunk.
    4. Paragraphs longer than chunk_size are further split by characters with overlap.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if not paragraphs:
        return []

    raw_chunks: list[str] = []
    buf_parts: list[str] = []
    buf_len = 0

    def _flush() -> None:
        if buf_parts:
            raw_chunks.append("\n\n".join(buf_parts))

    for para in paragraphs:
        if len(para) > chunk_size:
            # Flush accumulated buffer first
            _flush()
            buf_parts = []
            buf_len = 0
            # Character-split the oversized paragraph
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                raw_chunks.append(para[start:end])
                if end == len(para):
                    break
                start = end - overlap
        else:
            added_len = len(para) + (2 if buf_parts else 0)  # 2 for "\n\n"
            if buf_len + added_len > chunk_size and buf_parts:
                _flush()
                # Keep the last paragraph as overlap into the next chunk
                last = buf_parts[-1]
                buf_parts = [last, para]
                buf_len = len(last) + 2 + len(para)
            else:
                buf_parts.append(para)
                buf_len += added_len

    _flush()
    return raw_chunks
