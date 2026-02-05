"""PaddleOCR PP-Structure engine integration for Bookmind offline benchmarks."""

from __future__ import annotations

import json
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_PAGE_RE = re.compile(r"(?:page|pg|p)[_\- ]*(\d+)", re.IGNORECASE)


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._current_row: List[str] = []
        self._cell_chunks: List[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_chunks = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._in_cell = False
            cell_text = "".join(self._cell_chunks).strip()
            self._current_row.append(cell_text)
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_chunks.append(data)


def _page_num_from_name(path: Path) -> Optional[int]:
    match = _PAGE_RE.search(path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def list_page_images(img_dir: str) -> List[Tuple[int, Path]]:
    img_path = Path(img_dir)
    if not img_path.exists():
        raise FileNotFoundError(f"Image directory not found: {img_path}")

    images: List[Tuple[int, Path]] = []
    for path in sorted(img_path.glob("*.png")):
        page_num = _page_num_from_name(path)
        if page_num is not None:
            images.append((page_num, path))

    images.sort(key=lambda item: item[0])
    return images


def _ensure_cache_dir(cache_dir: str) -> Path:
    path = Path(cache_dir)
    if not path.exists():
        raise SystemExit(
            f"PaddleOCR cache directory not found: {path}. "
            "Run tools/bookmind_bench/scripts/prefetch_online.sh first."
        )
    if not any(path.iterdir()):
        raise SystemExit(
            f"PaddleOCR cache directory is empty: {path}. "
            "Run tools/bookmind_bench/scripts/prefetch_online.sh first."
        )
    return path


def _configure_cache_dir(cache_dir: Optional[str]) -> Optional[Path]:
    if not cache_dir:
        return None
    path = _ensure_cache_dir(cache_dir)
    os.environ["PADDLEOCR_HOME"] = str(path)
    os.environ["PADDLE_HOME"] = str(path)
    os.environ["HOME"] = str(path)
    return path


def _normalize_bbox(bbox: Any) -> Optional[List[float]]:
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)):
        if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            return [float(b) for b in bbox]
        if len(bbox) == 4 and all(
            isinstance(v, (list, tuple)) and len(v) == 2 for v in bbox
        ):
            xs = [float(v[0]) for v in bbox]
            ys = [float(v[1]) for v in bbox]
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _extract_text_from_res(res: Any) -> str:
    if res is None:
        return ""
    if isinstance(res, str):
        return res.strip()
    if isinstance(res, dict):
        if "text" in res and res["text"] is not None:
            return str(res["text"]).strip()
        return ""
    if isinstance(res, list):
        parts: List[str] = []
        for item in res:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p.strip() for p in parts if p.strip())
    return str(res).strip()


def _extract_confidence(res: Any, fallback: Optional[float]) -> Optional[float]:
    if isinstance(fallback, (int, float)):
        return float(fallback)
    if isinstance(res, list):
        values = [
            float(item["confidence"])
            for item in res
            if isinstance(item, dict)
            and isinstance(item.get("confidence"), (int, float))
        ]
        if values:
            return sum(values) / len(values)
    return None


def _html_table_to_markdown(html: str) -> str:
    if not html:
        return ""
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
    except Exception:
        return html.strip()

    rows = parser.rows
    if not rows:
        return html.strip()

    col_count = max(len(row) for row in rows)
    padded = [row + [""] * (col_count - len(row)) for row in rows]
    header = padded[0]
    separator = ["---"] * col_count
    body = padded[1:] if len(padded) > 1 else []

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_tokens_to_markdown(res: Any) -> str:
    if not isinstance(res, list):
        return ""
    texts: List[str] = []
    for item in res:
        if isinstance(item, dict) and item.get("text"):
            texts.append(str(item["text"]).strip())
        elif isinstance(item, str):
            texts.append(item.strip())
    texts = [t for t in texts if t]
    if not texts:
        return ""
    lines = [
        "| Cell |",
        "| --- |",
    ]
    for text in texts:
        lines.append(f"| {text} |")
    return "\n".join(lines)


def _normalize_block_type(raw_type: str) -> str:
    block_type = raw_type.strip().lower()
    if not block_type:
        return "unknown"
    mapping = {
        "text": "text",
        "title": "title",
        "list": "list",
        "table": "table",
        "figure": "figure",
        "equation": "equation",
    }
    return mapping.get(block_type, "unknown")


def _build_record(
    page: int,
    record_type: str,
    text: str,
    bbox: Optional[List[float]],
    timing_ms: int,
    block_type: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "engine": "paddleocr",
        "page": page,
        "type": record_type,
        "text": text,
        "timing_ms": timing_ms,
        "meta": {
            "pdf_page_start": page,
            "pdf_page_end": page,
        },
    }
    if bbox is not None:
        record["bbox"] = bbox
    if block_type:
        record["meta"]["block_type"] = block_type
    if confidence is not None:
        record["meta"]["confidence"] = confidence
    return record


def validate_paddleocr_record(record: Dict[str, Any]) -> None:
    if record.get("engine") != "paddleocr":
        raise ValueError("engine must be 'paddleocr'")
    if not isinstance(record.get("page"), int) or record["page"] < 1:
        raise ValueError("page must be a positive int")
    record_type = record.get("type")
    if record_type not in {"ocr_text", "table_md", "layout_block"}:
        raise ValueError("type must be ocr_text, table_md, or layout_block")
    if "text" not in record:
        raise ValueError("text is required")
    if record_type == "table_md" and not str(record.get("text", "")).strip():
        raise ValueError("table_md records must include text")
    meta = record.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dict")
    for key in ("pdf_page_start", "pdf_page_end"):
        if key not in meta:
            raise ValueError(f"meta.{key} is required")
    if record_type == "layout_block":
        if "bbox" not in record or record["bbox"] is None:
            raise ValueError("layout_block records must include bbox")
        if not meta.get("block_type"):
            raise ValueError("layout_block records must include meta.block_type")
    if "timing_ms" not in record:
        raise ValueError("timing_ms is required")
    if "bbox" in record and record["bbox"] is not None:
        bbox = record["bbox"]
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)
        ):
            raise ValueError("bbox must be [x1, y1, x2, y2]")


def run_paddleocr(
    img_dir: str,
    pages: Optional[Sequence[int]],
    lang: str,
    use_gpu: bool,
    cache_dir: Optional[str],
    debug_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cache_path = _configure_cache_dir(cache_dir)
    if cache_path is None:
        raise SystemExit(
            "PaddleOCR cache directory is not configured. "
            "Set BOOKMIND_PADDLEOCR_CACHE_DIR or run prefetch_online.sh first."
        )

    try:
        from paddleocr import PPStructure
    except Exception as exc:
        raise SystemExit(f"Unable to import PaddleOCR: {exc}")

    images = list_page_images(img_dir)
    if not images:
        raise SystemExit(f"No page images found in {img_dir}")

    if pages:
        page_set = set(pages)
        images = [item for item in images if item[0] in page_set]

    if not images:
        raise SystemExit("No matching pages found in image directory.")

    engine = PPStructure(show_log=False, use_gpu=use_gpu, lang=lang)

    records: List[Dict[str, Any]] = []
    debug_path: Optional[Path] = None
    if debug_dir:
        debug_path = Path(debug_dir) / "paddleocr_raw"
        debug_path.mkdir(parents=True, exist_ok=True)
    for page_num, image_path in images:
        start = time.perf_counter()
        try:
            result = engine(str(image_path))
        except Exception as exc:
            raise SystemExit(f"PaddleOCR failed on {image_path}: {exc}")
        timing_ms = int(round((time.perf_counter() - start) * 1000))
        if debug_path is not None:
            raw_path = debug_path / f"page_{page_num}.json"
            raw_path.write_text(json.dumps(result, ensure_ascii=True, indent=2))

        for item in result:
            if not isinstance(item, dict):
                continue
            block_type = _normalize_block_type(str(item.get("type", "")))
            bbox = _normalize_bbox(item.get("bbox"))
            res = item.get("res")
            confidence = _extract_confidence(res, item.get("confidence"))

            if block_type == "table":
                html = ""
                if isinstance(res, dict) and res.get("html"):
                    html = str(res["html"])
                if html:
                    text = _html_table_to_markdown(html)
                else:
                    text = _table_tokens_to_markdown(res)
                if not text:
                    text = "Table"
                records.append(
                    _build_record(
                        page=page_num,
                        record_type="table_md",
                        text=text,
                        bbox=bbox,
                        timing_ms=timing_ms,
                        block_type=block_type,
                        confidence=confidence,
                    )
                )
                continue

            text = _extract_text_from_res(res)
            if block_type == "figure":
                records.append(
                    _build_record(
                        page=page_num,
                        record_type="layout_block",
                        text="",
                        bbox=bbox,
                        timing_ms=timing_ms,
                        block_type=block_type,
                        confidence=confidence,
                    )
                )
                continue

            if text:
                record_type = "ocr_text"
            else:
                record_type = "layout_block"
                text = ""
            records.append(
                _build_record(
                    page=page_num,
                    record_type=record_type,
                    text=text,
                    bbox=bbox,
                    timing_ms=timing_ms,
                    block_type=block_type,
                    confidence=confidence,
                )
            )

    return records


def run_paddleocr_on_image(
    image_path: str,
    lang: str,
    use_gpu: bool,
    cache_dir: Optional[str],
) -> List[Dict[str, Any]]:
    image_path = str(image_path)
    cache_path = _configure_cache_dir(cache_dir)
    if cache_path is None:
        raise SystemExit(
            "PaddleOCR cache directory is not configured. "
            "Set BOOKMIND_PADDLEOCR_CACHE_DIR or run prefetch_online.sh first."
        )

    try:
        from paddleocr import PPStructure
    except Exception as exc:
        raise SystemExit(f"Unable to import PaddleOCR: {exc}")

    start = time.perf_counter()
    engine = PPStructure(show_log=False, use_gpu=use_gpu, lang=lang)
    result = engine(image_path)
    timing_ms = int(round((time.perf_counter() - start) * 1000))

    records: List[Dict[str, Any]] = []
    for item in result:
        if not isinstance(item, dict):
            continue
        block_type = _normalize_block_type(str(item.get("type", "")))
        bbox = _normalize_bbox(item.get("bbox"))
        res = item.get("res")
        confidence = _extract_confidence(res, item.get("confidence"))

        if block_type == "table":
            html = ""
            if isinstance(res, dict) and res.get("html"):
                html = str(res["html"])
            if html:
                text = _html_table_to_markdown(html)
            else:
                text = _table_tokens_to_markdown(res)
            if not text:
                text = "Table"
            records.append(
                _build_record(
                    page=1,
                    record_type="table_md",
                    text=text,
                    bbox=bbox,
                    timing_ms=timing_ms,
                    block_type=block_type,
                    confidence=confidence,
                )
            )
            continue

        text = _extract_text_from_res(res)
        if block_type == "figure":
            records.append(
                _build_record(
                    page=1,
                    record_type="layout_block",
                    text="",
                    bbox=bbox,
                    timing_ms=timing_ms,
                    block_type=block_type,
                    confidence=confidence,
                )
            )
            continue

        if text:
            record_type = "ocr_text"
        else:
            record_type = "layout_block"
            text = ""
        records.append(
            _build_record(
                page=1,
                record_type=record_type,
                text=text,
                bbox=bbox,
                timing_ms=timing_ms,
                block_type=block_type,
                confidence=confidence,
            )
        )

    return records
