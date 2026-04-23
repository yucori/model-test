"""
PDF / DOCX text extraction and chunking.

Supported parsers:
  - pdfplumber (default): fast, accurate for digital PDFs. Scanned pages fall back to easyocr.
  - pymupdf: PyMuPDF block-based extraction, different layout reconstruction from pdfplumber.
  - docling: IBM Docling with doctr backend. Preserves tables and headings as Markdown.
"""
import importlib.util
import os
import re
from pathlib import Path
from typing import Optional

# Disable PaddlePaddle 3.x PIR executor + oneDNN before any Paddle C++ code
# initialises. Must be set at module load time — setting inside a function is
# too late because Paddle reads these flags during C++ static initialisation.

# ── Common markdown post-processing ──────────────────────────────────────────


def _apply_heading_markdown(text: str) -> str:
    """Convert 제N장 / 제N조 patterns to markdown headings.

    Applies to pdfplumber and PyMuPDF output where structural tags exist as
    plain text.  Docling already emits markdown headings; DOCX uses style-based
    detection — neither needs this function.

    Rules:
      제N장  →  ##  (chapter, depth 2)
      제N조  →  ### (article, depth 3)
    Lines that already start with '#' are left untouched.
    """
    lines = text.split('\n')
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            result.append(line)
            continue
        if re.match(r'^제\s*\d+\s*장(\s|\(|$)', stripped):
            result.append('## ' + stripped)
        elif re.match(r'^제\s*\d+\s*조(\s|\(|$)', stripped):
            result.append('### ' + stripped)
        else:
            result.append(line)
    return '\n'.join(result)


# ── OCR state ─────────────────────────────────────────────────────────────────

_easyocr_reader = None
_OCR_TEXT_THRESHOLD = 50  # chars — below this a page is treated as image-only


# ── Parser registry ───────────────────────────────────────────────────────────

def _pkg_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def get_parser_infos() -> list[dict]:
    return [
        {
            "id": "pdfplumber",
            "name": "pdfplumber",
            "available": True,
            "file_types": ["pdf"],
            "description": "텍스트 레이어 정밀 추출. 스캔 페이지는 easyocr 자동 폴백.",
        },
        {
            "id": "pymupdf",
            "name": "PyMuPDF (fitz)",
            "available": _pkg_available("fitz"),
            "file_types": ["pdf"],
            "description": "빠른 텍스트·블록 추출. pdfplumber와 추출 방식이 달라 비교에 유용.",
        },
        {
            "id": "python-docx",
            "name": "python-docx",
            "available": _pkg_available("docx"),
            "file_types": ["docx"],
            "description": "DOCX 전용 파서. Word 헤딩 스타일 기반 구조 감지 + 표 마크다운 변환.",
        },
        {
            "id": "docling",
            "name": "Docling (IBM)",
            "available": _pkg_available("docling"),
            "file_types": ["pdf", "docx"],
            "description": "레이아웃 분석 + doctr OCR. 표·헤더 구조를 마크다운으로 보존.",
        },
    ]


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text(file_path: str, parser: str = "pdfplumber") -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        if parser == "pymupdf":
            return _extract_pymupdf(file_path)
        elif parser == "docling":
            return _extract_docling(file_path)
        else:
            return _extract_pdfplumber(file_path)
    elif ext in (".docx", ".doc"):
        if parser == "docling":
            return _extract_docling(file_path)
        else:  # python-docx (또는 잘못 넘어온 pdfplumber/pymupdf 폴백)
            return _extract_docx(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


# ── pdfplumber (default) ──────────────────────────────────────────────────────

def _pdfplumber_page_text(page) -> str:
    """Extract one page preserving reading order: tables as markdown, prose as plain text.

    Strategy:
      1. Find all table bounding boxes on the page.
      2. For each table, format its cells as a markdown table and record its
         top-Y position so we can interleave it correctly with surrounding prose.
      3. Filter the table regions OUT of the character stream, then extract the
         remaining words and group them into paragraphs by vertical gap.
      4. Sort ALL content blocks (tables + text paragraphs) by their top-Y
         coordinate so the original reading order is preserved exactly.
    """
    # ── Step 1+2: extract tables, record Y position ───────────────────────────
    content: list[tuple[float, str]] = []  # (top_y, text)
    table_bboxes: list[tuple] = []

    for tbl in page.find_tables():
        table_bboxes.append(tbl.bbox)
        data = tbl.extract()
        if not data or len(data) < 2:
            continue
        # 모든 셀이 비어있는 표는 텍스트 중복 추출된 것 — 건너뜀
        if all(not str(c or "").strip() for row in data for c in row):
            continue
        import pandas as pd
        df = pd.DataFrame(data[1:], columns=data[0])
        content.append((tbl.bbox[1], df.to_markdown(index=False)))

    # ── Step 3: extract non-table words ──────────────────────────────────────
    if table_bboxes:
        def _outside_tables(obj):
            for x0, top, x1, bot in table_bboxes:
                if (obj.get("x0", 0) >= x0 - 3 and obj.get("x1", 0) <= x1 + 3 and
                        obj.get("top", 0) >= top - 3 and obj.get("bottom", 0) <= bot + 3):
                    return False
            return True
        words = page.filter(_outside_tables).extract_words(keep_blank_chars=False, use_text_flow=True)
    else:
        words = page.extract_words(keep_blank_chars=False, use_text_flow=True)

    # Group words → lines (same baseline ±4pt), then lines → paragraphs (gap >12pt)
    if words:
        lines: list[tuple[float, str]] = []
        cur_top: float | None = None
        cur_words: list[dict] = []

        for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
            if cur_top is None or abs(w["top"] - cur_top) <= 4:
                cur_words.append(w)
                cur_top = w["top"] if cur_top is None else min(cur_top, w["top"])
            else:
                if cur_words:
                    t = min(ww["top"] for ww in cur_words)
                    sorted_words = sorted(cur_words, key=lambda ww: ww["x0"])
                    lines.append((t, " ".join(ww["text"] for ww in sorted_words)))
                cur_words = [w]
                cur_top = w["top"]
        if cur_words:
            t = min(ww["top"] for ww in cur_words)
            sorted_words = sorted(cur_words, key=lambda ww: ww["x0"])
            lines.append((t, " ".join(ww["text"] for ww in sorted_words)))

        # Merge lines into paragraphs when gap ≤ 12pt
        para_lines: list[tuple[float, str]] = [lines[0]] if lines else []
        for i in range(1, len(lines)):
            gap = lines[i][0] - lines[i - 1][0]
            if gap <= 12:
                para_lines.append(lines[i])
            else:
                if para_lines:
                    content.append((para_lines[0][0], "\n".join(l[1] for l in para_lines)))
                para_lines = [lines[i]]
        if para_lines:
            content.append((para_lines[0][0], "\n".join(l[1] for l in para_lines)))

    # ── Step 4: sort by Y position → original reading order ──────────────────
    content.sort(key=lambda x: x[0])
    return "\n\n".join(text for _, text in content)


def _extract_pdfplumber(path: str) -> str:
    import pdfplumber

    pages_text: list[str] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = _pdfplumber_page_text(page)
            if len(text) >= _OCR_TEXT_THRESHOLD:
                pages_text.append(text)
            else:
                ocr_text = _ocr_page_easyocr(path, i)
                pages_text.append(ocr_text if ocr_text else text)
    full_text = "\n\n".join(filter(None, pages_text))
    return _apply_heading_markdown(full_text)


def _ocr_page_easyocr(pdf_path: str, page_index: int) -> Optional[str]:
    """Render one PDF page and extract text with easyocr."""
    global _easyocr_reader
    try:
        import fitz
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
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        results = _easyocr_reader.readtext(img_bytes, detail=0, paragraph=True)
        return "\n".join(results).strip() or None
    except Exception as e:
        print(f"[OCR] easyocr failed on page {page_index} of {pdf_path}: {e}")
        return None


# ── PyMuPDF ──────────────────────────────────────────────────────────────────

def _extract_pymupdf(path: str) -> str:
    """Extract text using PyMuPDF (fitz).

    Tables are detected via page.find_tables() (PyMuPDF 1.23+) and rendered
    as markdown.  Non-table text blocks are extracted from get_text("dict")
    with table regions excluded.  All content is sorted by Y position to
    preserve reading order.  Header/footer zones (top/bottom 7%) are stripped.
    """
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF가 설치되어 있지 않습니다. 설치: uv add pymupdf")
    doc = fitz.open(path)
    pages_text: list[str] = []

    for page in doc:
        h = page.rect.height
        margin = h * 0.07
        content: list[tuple[float, str]] = []  # (top_y, text)
        table_bboxes: list[tuple] = []

        # ── Tables → markdown ─────────────────────────────────────────────────
        if hasattr(page, "find_tables"):
            import pandas as pd
            for tab in page.find_tables():
                table_bboxes.append(tab.bbox)
                df = tab.to_pandas()
                # 모든 셀이 비어있는 표(장식용 테두리 등)는 건너뜀
                if df.empty or all(
                    not str(v).strip() or str(v) in ("nan", "None")
                    for v in df.values.flatten()
                ):
                    continue
                content.append((tab.bbox[1], df.to_markdown(index=False)))

        # ── Text blocks (table regions excluded) ──────────────────────────────
        blocks = page.get_text("dict", sort=True)["blocks"]
        for blk in blocks:
            if blk.get("type") != 0:
                continue
            bx0, btop, bx1, bbot = blk["bbox"]
            if bbot < margin or btop > h - margin:
                continue
            # Skip cells already captured as table markdown
            if any(
                bx0 >= tx0 - 1 and bx1 <= tx1 + 1 and btop >= ttop - 1 and bbot <= tbot + 1
                for tx0, ttop, tx1, tbot in table_bboxes
            ):
                continue
            lines = []
            for line in blk.get("lines", []):
                line_text = "".join(span["text"] for span in line.get("spans", [])).strip()
                if line_text:
                    lines.append(line_text)
            block_text = "\n".join(lines).strip()
            if block_text:
                content.append((btop, block_text))

        if content:
            content.sort(key=lambda x: x[0])
            pages_text.append("\n\n".join(t for _, t in content))

    full_text = "\n\n".join(pages_text)
    return _apply_heading_markdown(full_text)


# ── Docling ───────────────────────────────────────────────────────────────────

def _patch_cv2() -> None:
    """Replace a broken/stub cv2 with a working shim backed by numpy + Pillow.

    The installed cv2 is a type-stub package (opencv-stubs or similar) that has
    no actual implementation — not just missing constants, but missing core
    functions like resize().  We detect this early and swap in PIL/numpy
    implementations for every function docling/doctr calls.
    """
    import sys
    import types
    import numpy as np
    from PIL import Image as _PIL

    import cv2 as _cv2_mod

    # If resize exists we have a real OpenCV — only patch minor gaps.
    _real_cv2 = hasattr(_cv2_mod, "resize")

    if _real_cv2:
        if not hasattr(_cv2_mod, "setNumThreads"):
            _cv2_mod.setNumThreads = lambda n: None  # type: ignore[attr-defined]
        if not hasattr(_cv2_mod, "ocl"):
            class _OclStub:
                def setUseOpenCL(self, f: bool) -> None: ...
                def useOpenCL(self) -> bool: return False
            _cv2_mod.ocl = _OclStub()  # type: ignore[attr-defined]
        return

    # ── Build a full shim module ──────────────────────────────────────────────
    shim = types.ModuleType("cv2")
    shim.__spec__ = _cv2_mod.__spec__

    # ── Constants ────────────────────────────────────────────────────────────
    _CONSTANTS: dict[str, int] = {
        # interpolation
        "INTER_NEAREST": 0, "INTER_LINEAR": 1, "INTER_CUBIC": 2,
        "INTER_AREA": 3, "INTER_LANCZOS4": 4,
        "INTER_LINEAR_EXACT": 5, "INTER_NEAREST_EXACT": 6,
        # border
        "BORDER_CONSTANT": 0, "BORDER_REPLICATE": 1, "BORDER_REFLECT": 2,
        "BORDER_WRAP": 3, "BORDER_REFLECT_101": 4, "BORDER_REFLECT101": 4,
        "BORDER_DEFAULT": 4, "BORDER_TRANSPARENT": 5, "BORDER_ISOLATED": 16,
        # color
        "COLOR_BGR2BGRA": 0, "COLOR_BGRA2BGR": 1,
        "COLOR_BGR2RGB": 4, "COLOR_RGB2BGR": 4,
        "COLOR_BGRA2RGBA": 5, "COLOR_RGBA2BGRA": 5,
        "COLOR_BGR2GRAY": 6, "COLOR_GRAY2BGR": 8, "COLOR_GRAY2BGRA": 9,
        "COLOR_BGRA2GRAY": 10, "COLOR_RGBA2GRAY": 11,
        "COLOR_BGR2HSV": 40, "COLOR_RGB2HSV": 41,
        "COLOR_HSV2BGR": 54, "COLOR_HSV2RGB": 55,
        "COLOR_BGR2Lab": 44, "COLOR_RGB2Lab": 45,
        "COLOR_Lab2BGR": 56, "COLOR_Lab2RGB": 57,
        "COLOR_BGR2YCrCb": 36, "COLOR_YCrCb2BGR": 38,
        # threshold
        "THRESH_BINARY": 0, "THRESH_BINARY_INV": 1, "THRESH_TRUNC": 2,
        "THRESH_TOZERO": 3, "THRESH_TOZERO_INV": 4,
        "THRESH_OTSU": 8, "THRESH_TRIANGLE": 16,
        # morphology
        "MORPH_RECT": 0, "MORPH_CROSS": 1, "MORPH_ELLIPSE": 2,
        "MORPH_ERODE": 0, "MORPH_DILATE": 1, "MORPH_OPEN": 2,
        "MORPH_CLOSE": 3, "MORPH_GRADIENT": 4,
        # contours
        "RETR_EXTERNAL": 0, "RETR_LIST": 1, "RETR_CCOMP": 2, "RETR_TREE": 3,
        "CHAIN_APPROX_NONE": 1, "CHAIN_APPROX_SIMPLE": 2,
        # drawing
        "LINE_4": 4, "LINE_8": 8, "LINE_AA": 16, "FILLED": -1,
        "FONT_HERSHEY_SIMPLEX": 0, "FONT_HERSHEY_PLAIN": 1,
        # misc
        "CAP_PROP_FRAME_COUNT": 7, "CAP_PROP_FPS": 5,
        "NORM_MINMAX": 32, "NORM_L2": 4,
    }
    for k, v in _CONSTANTS.items():
        setattr(shim, k, v)

    # ── ocl stub ─────────────────────────────────────────────────────────────
    class _OclStub:
        def setUseOpenCL(self, f: bool) -> None: ...
        def useOpenCL(self) -> bool: return False
    shim.ocl = _OclStub()  # type: ignore[attr-defined]
    shim.setNumThreads = lambda n: None  # type: ignore[attr-defined]

    # ── Helper: map PIL resample from cv2 interpolation flag ─────────────────
    _RESAMPLE = {0: _PIL.NEAREST, 1: _PIL.BILINEAR, 2: _PIL.BICUBIC,
                 3: _PIL.BOX, 4: _PIL.LANCZOS}

    def _arr(src):
        return np.asarray(src)

    def _pil(arr):
        a = np.asarray(arr)
        if a.ndim == 2:
            return _PIL.fromarray(a.astype(np.uint8), "L")
        if a.shape[2] == 4:
            return _PIL.fromarray(a.astype(np.uint8), "RGBA")
        return _PIL.fromarray(a.astype(np.uint8), "RGB")

    # ── Core image functions ──────────────────────────────────────────────────

    def resize(src, dsize, dst=None, fx=0.0, fy=0.0, interpolation=1):
        arr = _arr(src)
        h, w = arr.shape[:2]
        if dsize and dsize[0] > 0 and dsize[1] > 0:
            nw, nh = int(dsize[0]), int(dsize[1])
        else:
            nw, nh = max(1, int(w * fx)), max(1, int(h * fy))
        rsmp = _RESAMPLE.get(interpolation, _PIL.BILINEAR)
        return np.array(_pil(arr).resize((nw, nh), rsmp))

    _COLOR_CONV = {
        4:  lambda a: a[:, :, ::-1],           # BGR2RGB / RGB2BGR
        6:  lambda a: np.array(_pil(a).convert("L")),  # BGR2GRAY
        8:  lambda a: np.stack([np.array(_pil(a).convert("L"))] * 3, axis=2),  # GRAY2BGR
        10: lambda a: np.array(_pil(a[:, :, :3]).convert("L")),  # BGRA2GRAY
        11: lambda a: np.array(_pil(a[:, :, :3]).convert("L")),  # RGBA2GRAY
        9:  lambda a: np.dstack([np.array(_pil(a).convert("L"))] * 3 + [np.full(a.shape[:2], 255, np.uint8)]),
    }

    def cvtColor(src, code, dst=None, dstCn=0):
        arr = _arr(src)
        fn = _COLOR_CONV.get(code)
        if fn:
            return fn(arr).astype(np.uint8)
        return arr.copy()

    def imencode(ext, img, params=None):
        import io
        buf = io.BytesIO()
        fmt = ext.lstrip(".").upper()
        if fmt == "JPG":
            fmt = "JPEG"
        _pil(_arr(img)).save(buf, format=fmt)
        return True, np.frombuffer(buf.getvalue(), dtype=np.uint8)

    def imdecode(buf, flags=1):
        import io
        data = bytes(buf) if not isinstance(buf, (bytes, bytearray)) else buf
        img = _PIL.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)
        if flags == 0:  # IMREAD_GRAYSCALE
            arr = np.array(img.convert("L"))
        return arr[:, :, ::-1].astype(np.uint8) if arr.ndim == 3 else arr

    def threshold(src, thresh, maxval, type_):
        arr = _arr(src).astype(np.uint8)
        binary = (arr > thresh).astype(np.uint8) * int(maxval)
        if type_ == 1:  # THRESH_BINARY_INV
            binary = int(maxval) - binary
        return thresh, binary

    def GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=0, borderType=4):
        from PIL import ImageFilter
        arr = _arr(src)
        r = max(1, ksize[0] // 2)
        return np.array(_pil(arr).filter(ImageFilter.GaussianBlur(radius=r)))

    def getStructuringElement(shape, ksize, anchor=(-1, -1)):
        return np.ones((ksize[1], ksize[0]), dtype=np.uint8)

    def _morph_op(src, kernel, iters, dilate: bool):
        from scipy.ndimage import binary_dilation, binary_erosion
        arr = _arr(src).astype(bool)
        fn = binary_dilation if dilate else binary_erosion
        for _ in range(max(1, iters)):
            arr = fn(arr, structure=kernel.astype(bool))
        return arr.astype(np.uint8) * 255

    def dilate(src, kernel, dst=None, anchor=(-1,-1), iterations=1, borderType=4, borderValue=0):
        return _morph_op(src, kernel, iterations, True)

    def erode(src, kernel, dst=None, anchor=(-1,-1), iterations=1, borderType=4, borderValue=0):
        return _morph_op(src, kernel, iterations, False)

    def findContours(image, mode, method, contours=None, hierarchy=None, offset=(0,0)):
        # Return empty contours — used for layout analysis heuristics
        return [], np.array([])

    def boundingRect(contour):
        if len(contour) == 0:
            return 0, 0, 0, 0
        pts = np.asarray(contour).reshape(-1, 2)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        w, h = pts[:, 0].max() - x, pts[:, 1].max() - y
        return int(x), int(y), int(w), int(h)

    def connectedComponents(src, connectivity=8, ltype=4):
        from scipy.ndimage import label
        arr = (_arr(src) > 0).astype(np.int32)
        labeled, n = label(arr)
        return n + 1, labeled.astype(np.int32)

    def connectedComponentsWithStats(src, connectivity=8, ltype=4):
        n, labels = connectedComponents(src, connectivity, ltype)
        stats = np.zeros((n, 5), dtype=np.int32)
        centroids = np.zeros((n, 2), dtype=np.float64)
        return n, labels, stats, centroids

    def copyMakeBorder(src, top, bottom, left, right, borderType=0, value=0):
        from PIL import ImageOps
        arr = _arr(src)
        if arr.ndim == 2:
            img = _PIL.fromarray(arr.astype(np.uint8), "L")
        else:
            img = _pil(arr)
        padded = ImageOps.expand(img, (left, top, right, bottom), fill=value)
        return np.array(padded)

    def normalize(src, dst=None, alpha=1.0, beta=0.0, norm_type=32, dtype=-1, mask=None):
        arr = _arr(src).astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * (alpha - beta) + beta
        return arr

    def warpAffine(src, M, dsize, dst=None, flags=1, borderMode=0, borderValue=0):
        arr = _arr(src)
        img = _pil(arr)
        # PIL affine takes inverse transform; approximate with resize to dsize
        return np.array(img.resize((dsize[0], dsize[1]), _PIL.BILINEAR))

    def rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0):
        arr = _arr(img).copy()
        x1, y1 = max(0, pt1[0]), max(0, pt1[1])
        x2, y2 = min(arr.shape[1]-1, pt2[0]), min(arr.shape[0]-1, pt2[1])
        if thickness == -1:
            arr[y1:y2, x1:x2] = color
        else:
            t = max(1, thickness)
            arr[y1:y1+t, x1:x2] = color
            arr[y2-t:y2, x1:x2] = color
            arr[y1:y2, x1:x1+t] = color
            arr[y1:y2, x2-t:x2] = color
        return arr

    def putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType=8, bottomLeftOrigin=False):
        return _arr(img).copy()  # text rendering not needed for extraction

    def getRotationMatrix2D(center, angle, scale):
        import math
        cx, cy = center
        a = math.radians(angle)
        c, s = math.cos(a) * scale, math.sin(a) * scale
        return np.array([[c, s, (1-c)*cx - s*cy],
                         [-s, c, s*cx + (1-c)*cy]], dtype=np.float64)

    for _fn in [
        resize, cvtColor, imencode, imdecode, threshold, GaussianBlur,
        getStructuringElement, dilate, erode, findContours, boundingRect,
        connectedComponents, connectedComponentsWithStats, copyMakeBorder,
        normalize, warpAffine, rectangle, putText, getRotationMatrix2D,
    ]:
        setattr(shim, _fn.__name__, _fn)

    # Replace the broken module everywhere it may already be cached
    sys.modules["cv2"] = shim


def _extract_docling(path: str) -> str:
    """Extract text with Docling (IBM).

    Speed optimisations applied:
      - do_ocr=False  — digital PDFs already have a text layer; skip the
                        heavy doctr OCR pipeline (saves 80-90% of processing time).
      - do_table_structure=True — keep table structure recognition since that is
                                   the main reason to use Docling over pdfplumber.
    If the file is a scanned PDF (no text layer), re-raise with a helpful message.
    """
    try:
        _patch_cv2()
        from docling.document_converter import DocumentConverter, InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption
    except ImportError:
        raise RuntimeError("Docling이 설치되어 있지 않습니다. 설치: uv add docling")
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False            # skip OCR — digital PDF only
        pipeline_options.do_table_structure = True  # keep table parsing (Docling's strength)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        result = converter.convert(path)
        return result.document.export_to_markdown()
    except Exception as e:
        raise RuntimeError(f"Docling 추출 실패: {e}")


# ── PaddleOCR ─────────────────────────────────────────────────────────────────


# ── DOCX ──────────────────────────────────────────────────────────────────────

def _extract_docx(path: str) -> str:
    """Extract text from DOCX preserving document structure as Markdown.

    Strategy:
      - Iterate body children in order (paragraphs + tables interleaved) so
        reading order is preserved exactly.
      - Paragraph style names drive heading level:
          'Heading 1' → ##   'Heading 2' → ###   'Heading 3' → ####
      - Paragraphs without a heading style but with bold text matching the
        제N조 / 제N장 pattern are promoted to ### / ##.
      - Tables are rendered as GitHub-flavoured markdown tables.
    """
    from docx import Document
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    from docx.table import Table as DocxTable

    doc = Document(path)
    PARA_TAG = qn("w:p")
    TBL_TAG = qn("w:tbl")

    def _para_to_md(para: Paragraph) -> str:
        text = para.text.strip()
        if not text:
            return ""
        style = para.style.name if para.style else ""
        # Heading styles → markdown levels
        if "Heading 1" in style:
            return f"## {text}"
        if "Heading 2" in style:
            return f"### {text}"
        if "Heading 3" in style:
            return f"#### {text}"
        # Bold paragraph matching 장/조 pattern (fallback for docs without heading styles)
        is_bold = any(run.bold for run in para.runs if run.text.strip())
        if is_bold:
            if re.match(r'^제\s*\d+\s*장(\s|\(|$)', text):
                return f"## {text}"
            if re.match(r'^제\s*\d+\s*조(\s|\(|$)', text):
                return f"### {text}"
        return text

    def _table_to_md(table: DocxTable) -> str:
        import pandas as pd
        rows: list[list[str]] = []
        for row in table.rows:
            seen: set[int] = set()
            cells: list[str] = []
            for cell in row.cells:
                if id(cell._tc) not in seen:
                    seen.add(id(cell._tc))
                    cells.append(cell.text.strip().replace("\n", " "))
            rows.append(cells)
        if len(rows) < 2:
            return ""
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df.to_markdown(index=False)

    parts: list[str] = []
    for child in doc.element.body:
        if child.tag == PARA_TAG:
            md = _para_to_md(Paragraph(child, doc))
            if md:
                parts.append(md)
        elif child.tag == TBL_TAG:
            parts.append(_table_to_md(DocxTable(child, doc)))

    return "\n\n".join(parts)


# ── Chunking ──────────────────────────────────────────────────────────────────

CHUNK_STRATEGIES = [
    {
        "id": "paragraph",
        "name": "문단 단위",
        "description": "빈 줄 기준 자연스러운 문단 경계 우선. 균형 잡힌 범용 전략.",
        "supports_overlap": True,
    },
    {
        "id": "sentence",
        "name": "문장 단위",
        "description": "마침표·줄바꿈 기준 문장 분리 후 chunk_size까지 합산. 짧고 정밀한 청크.",
        "supports_overlap": False,
    },
    {
        "id": "fixed",
        "name": "고정 크기",
        "description": "글자 수 기준 슬라이딩 윈도우. 단순하고 예측 가능. 구조 무관.",
        "supports_overlap": True,
    },
    {
        "id": "semantic",
        "name": "시멘틱",
        "description": "문장 임베딩 유사도로 의미 경계 감지. 가장 정교하지만 처리 시간이 걸림.",
        "supports_overlap": False,
    },
    {
        "id": "parent_child",
        "name": "Parent-Child",
        "description": "큰 부모 청크(~1200자)를 작은 자식 청크(chunk_size)로 분할. "
                       "검색은 자식으로(정밀), LLM 컨텍스트는 부모로(풍부).",
        "supports_overlap": False,
    },
    {
        "id": "structure",
        "name": "구조 단위",
        "description": "## 장 / ### 조 마크다운 헤딩 기준으로 분할. "
                       "제N조 단위로 청크가 시작되어 정책 문서에 최적.",
        "supports_overlap": False,
    },
]


def get_chunk_strategy_infos() -> list[dict]:
    return CHUNK_STRATEGIES


def chunk_text(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100,
    strategy: str = "paragraph",
) -> list[str]:
    """Split text into chunks using the specified strategy.

    For parent_child strategy, returns child chunks (small).
    Use chunk_text_parent_child() to get both children and parent mapping.

    strategy options:
      - "paragraph" (default): blank-line boundaries, greedy merge up to chunk_size
      - "sentence": sentence-boundary split, then merge up to chunk_size
      - "fixed": character sliding window with overlap
      - "semantic": sentence embeddings + cosine-similarity boundary detection
      - "parent_child": parent=1200-char, child=chunk_size child chunks
    """
    text = text.strip()
    if not text:
        return []

    if strategy == "sentence":
        return _chunk_sentence(text, chunk_size)
    elif strategy == "fixed":
        return _chunk_fixed(text, chunk_size, overlap)
    elif strategy == "semantic":
        return _chunk_semantic(text, chunk_size)
    elif strategy == "parent_child":
        pc = chunk_text_parent_child(text, child_size=chunk_size)
        return pc["children"]
    elif strategy == "structure":
        return _chunk_structure(text, max_size=chunk_size)
    else:
        return _chunk_paragraph(text, chunk_size, overlap)


# ── Strategy implementations ──────────────────────────────────────────────────

def _chunk_paragraph(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Original paragraph-based chunker."""
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
            _flush()
            buf_parts = []
            buf_len = 0
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                raw_chunks.append(para[start:end])
                if end == len(para):
                    break
                start = end - overlap
        else:
            added_len = len(para) + (2 if buf_parts else 0)
            if buf_len + added_len > chunk_size and buf_parts:
                _flush()
                last = buf_parts[-1]
                buf_parts = [last, para]
                buf_len = len(last) + 2 + len(para)
            else:
                buf_parts.append(para)
                buf_len += added_len

    _flush()
    return raw_chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on Korean/English sentence-ending punctuation."""
    # Split on . ! ? followed by whitespace or end, and on newlines
    parts = re.split(r'(?<=[.!?。！？])\s+|\n+', text)
    return [s.strip() for s in parts if s.strip()]


def _chunk_sentence(text: str, chunk_size: int) -> list[str]:
    """Merge sentences greedily up to chunk_size."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for sent in sentences:
        added = len(sent) + (1 if buf else 0)
        if buf_len + added > chunk_size and buf:
            chunks.append(" ".join(buf))
            buf = [sent]
            buf_len = len(sent)
        else:
            buf.append(sent)
            buf_len += added

    if buf:
        chunks.append(" ".join(buf))

    return chunks


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window chunker."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_text_parent_child(
    text: str,
    child_size: int = 300,
    parent_size: int = 1200,
) -> dict:
    """Split text into parent-child chunk pairs.

    Returns a dict:
      {
        "children": list[str],       # small child chunks (indexed in ChromaDB)
        "parents": list[str],        # large parent chunks (used as LLM context)
        "child_to_parent": list[int] # index of parent for each child
      }

    Strategy:
      1. Split text into parent chunks (paragraph-based, ~parent_size chars)
      2. Split each parent into child chunks (fixed, ~child_size chars)
    """
    text = text.strip()
    if not text:
        return {"children": [], "parents": [], "child_to_parent": []}

    parents = _chunk_paragraph(text, parent_size, overlap=0)
    children: list[str] = []
    child_to_parent: list[int] = []

    for p_idx, parent in enumerate(parents):
        if len(parent) <= child_size:
            children.append(parent)
            child_to_parent.append(p_idx)
        else:
            child_chunks = _chunk_fixed(parent, child_size, overlap=0)
            for child in child_chunks:
                children.append(child)
                child_to_parent.append(p_idx)

    return {
        "children": children,
        "parents": parents,
        "child_to_parent": child_to_parent,
    }


def _chunk_structure(text: str, max_size: int = 800) -> list[str]:
    """Split on ## / ### markdown heading boundaries (제N장 / 제N조).

    Requires text that already has heading markers applied:
      - PDF: _apply_heading_markdown() converts 제N장/조 to ## / ###
      - DOCX: _extract_docx() emits ## / ### from Word heading styles

    Each heading and its following content form one chunk.
    Sections exceeding max_size are further split by paragraph (no overlap).
    """
    # Split just before every ## or ### heading, keeping the heading in its section
    parts = re.split(r'(?=^#{2,3} )', text, flags=re.MULTILINE)
    chunks: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_size:
            chunks.append(part)
        else:
            # Section too long — split by paragraph, first sub-chunk keeps the heading
            sub = _chunk_paragraph(part, max_size, overlap=0)
            chunks.extend(sub)
    return [c for c in chunks if c.strip()]


def _chunk_semantic(text: str, chunk_size: int, similarity_threshold: float = 0.5) -> list[str]:
    """Group sentences by embedding similarity.

    Uses the always-available local all-MiniLM-L6-v2 (ONNX) model via ChromaDB.
    Falls back to paragraph chunking if the model is unavailable.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return sentences if sentences else []

    try:
        import numpy as np
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            ef = DefaultEmbeddingFunction()
        except Exception:
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
            ef = ONNXMiniLM_L6_V2()

        embeddings = np.array(ef(sentences), dtype=np.float32)
        # Normalise rows for fast cosine via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embeddings = embeddings / norms

        chunks: list[str] = []
        buf = [sentences[0]]
        buf_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            added = len(sentences[i]) + 1
            # New chunk boundary: meaning shifts OR buffer would overflow chunk_size
            if sim < similarity_threshold or buf_len + added > chunk_size:
                chunks.append(" ".join(buf))
                buf = [sentences[i]]
                buf_len = len(sentences[i])
            else:
                buf.append(sentences[i])
                buf_len += added

        if buf:
            chunks.append(" ".join(buf))

        return chunks

    except Exception as e:
        print(f"[semantic chunker] fallback to paragraph: {e}")
        return _chunk_paragraph(text, chunk_size, overlap=0)
