"""Read source text and rasterize a PDF page for source or artifact review."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from threading import RLock

import pypdfium2 as pdfium

# PDFium shares native process-global state. Serialize access in the web process.
_LOCK = RLock()


def read_pages(path: Path) -> list[str]:
    """Extract page text in physical page order, including empty scanned pages."""
    with _LOCK, pdfium.PdfDocument(path) as document:
        pages = []
        for page in document:
            text_page = page.get_textpage()
            try:
                pages.append(text_page.get_text_range().replace("\r\n", "\n"))
            finally:
                text_page.close()
                page.close()
        return pages


def page_count(path: Path) -> int:
    with _LOCK, pdfium.PdfDocument(path) as document:
        return len(document)


def render_page(path: Path, number: int, scale: float = 1.5) -> bytes:
    """Render one physical page to PNG. Public page numbers start at one."""
    with _LOCK, pdfium.PdfDocument(path) as document:
        if not 1 <= number <= len(document):
            raise ValueError(f"Page must be between 1 and {len(document)}.")
        page = document[number - 1]
        try:
            width, height = page.get_size()
            bitmap = page.render(scale=min(scale, 2400 / max(width, height)))
            try:
                stream = BytesIO()
                bitmap.to_pil().save(stream, format="PNG")
                return stream.getvalue()
            finally:
                bitmap.close()
        finally:
            page.close()

