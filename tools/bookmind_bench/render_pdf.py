"""PDF rendering utilities for Bookmind benchmarks (offline)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import fitz  # PyMuPDF


@dataclass(frozen=True)
class RenderResult:
    pdf_path: Path
    output_dir: Path
    pages: List[int]
    dpi: int
    image_paths: List[Path]


def parse_pages(pages: str, max_page: int) -> List[int]:
    """Parse a 1-based page range string like "1-3,5,7-9"."""
    pages = pages.strip()
    if not pages:
        return []

    selected: List[int] = []
    parts = [p.strip() for p in pages.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_s, end_s = [p.strip() for p in part.split("-", 1)]
            start = int(start_s)
            end = int(end_s)
            if start <= 0 or end <= 0:
                raise ValueError("Page numbers must be >= 1")
            if start > end:
                raise ValueError(f"Invalid page range: {part}")
            selected.extend(range(start, end + 1))
        else:
            page = int(part)
            if page <= 0:
                raise ValueError("Page numbers must be >= 1")
            selected.append(page)

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for page in selected:
        if page not in seen:
            deduped.append(page)
            seen.add(page)

    # Clamp to max pages
    return [p for p in deduped if p <= max_page]


def render_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    pages: Iterable[int],
    dpi: int = 300,
) -> RenderResult:
    """Render selected pages of a PDF to PNG images at the given DPI."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dpi <= 0:
        raise ValueError("dpi must be > 0")

    with fitz.open(pdf_path) as doc:
        max_page = doc.page_count
        pages = [p for p in pages if 1 <= p <= max_page]
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        image_paths: List[Path] = []
        for page_number in pages:
            page = doc.load_page(page_number - 1)  # zero-based
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = output_dir / f"page_{page_number:04d}.png"
            pix.save(out_path.as_posix())
            image_paths.append(out_path)

    return RenderResult(
        pdf_path=pdf_path,
        output_dir=output_dir,
        pages=list(pages),
        dpi=dpi,
        image_paths=image_paths,
    )
