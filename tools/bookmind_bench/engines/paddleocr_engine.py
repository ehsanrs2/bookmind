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
FULLPAGE_FIGURE_AREA_RATIO = 0.80
MIN_TEXT_CHARS = 2
MAX_TEXT_BLOCKS_PER_PAGE = 500


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


def _bbox_area(bbox: Sequence[float]) -> float:
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    return width * height


def _page_area(image_path: Path) -> Optional[float]:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return float(width * height)


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


def _is_ocr_line(item: Any) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return False
    if not isinstance(item[0], (list, tuple)):
        return False
    if _normalize_bbox(item[0]) is None:
        return False
    if not isinstance(item[1], (list, tuple)) or not item[1]:
        return False
    return True


def _is_ocr_points(points: Any) -> bool:
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        return False
    return all(
        isinstance(point, (list, tuple))
        and len(point) == 2
        and all(isinstance(v, (int, float)) for v in point)
        for point in points
    )


def _normalize_ocr_lines_from_list(ocr_result: List[Any]) -> List[Any]:
    if ocr_result and all(_is_ocr_line(item) for item in ocr_result):
        return ocr_result
    if (
        len(ocr_result) == 1
        and isinstance(ocr_result[0], list)
        and ocr_result[0]
        and all(_is_ocr_line(item) for item in ocr_result[0])
    ):
        return ocr_result[0]
    if (
        len(ocr_result) == 1
        and isinstance(ocr_result[0], list)
        and ocr_result[0]
        and all(isinstance(item, (list, tuple)) and len(item) >= 2 for item in ocr_result[0])
    ):
        return ocr_result[0]
    return []


def _normalize_ocr_from_dict(ocr_result: Dict[str, Any]) -> List[Any]:
    def _maybe_list(value: Any) -> Any:
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                return value
        return value

    if "boxes" in ocr_result and "texts" in ocr_result:
        boxes = _maybe_list(ocr_result.get("boxes") or [])
        texts = ocr_result.get("texts") or []
        scores = ocr_result.get("scores") or ocr_result.get("confidences") or []
        lines: List[Any] = []
        for idx, box in enumerate(boxes):
            text = texts[idx] if idx < len(texts) else ""
            score = scores[idx] if idx < len(scores) else None
            lines.append([box, (text, score)])
        return lines
    if "dt_boxes" in ocr_result and "rec_res" in ocr_result:
        boxes = _maybe_list(ocr_result.get("dt_boxes") or [])
        rec_res = ocr_result.get("rec_res") or []
        lines = []
        for idx, box in enumerate(boxes):
            text = ""
            score = None
            if idx < len(rec_res):
                entry = rec_res[idx]
                if isinstance(entry, (list, tuple)) and entry:
                    text = entry[0]
                    if len(entry) > 1:
                        score = entry[1]
                elif isinstance(entry, dict):
                    text = entry.get("text", "")
                    score = entry.get("confidence")
            lines.append([box, (text, score)])
        return lines
    if "result" in ocr_result and isinstance(ocr_result["result"], list):
        return _normalize_ocr_lines(ocr_result["result"])
    return []


def _normalize_ocr_lines(ocr_result: Any) -> List[Any]:
    if isinstance(ocr_result, dict):
        return _normalize_ocr_from_dict(ocr_result)
    if isinstance(ocr_result, list):
        lines = _normalize_ocr_lines_from_list(ocr_result)
        if lines:
            return lines
        if ocr_result and all(isinstance(item, dict) for item in ocr_result):
            lines = []
            for item in ocr_result:
                if "bbox" in item and "text" in item:
                    lines.append(
                        [
                            item["bbox"],
                            (item.get("text"), item.get("confidence")),
                        ]
                    )
            if lines:
                return lines
    return []


def _extract_ocr_text_blocks(
    ocr_result: Any,
) -> Tuple[List[Tuple[List[float], str, Optional[float]]], int, int, int]:
    blocks: List[Tuple[List[float], str, Optional[float]]] = []
    short_filtered = 0
    lines = _normalize_ocr_lines(ocr_result)
    raw_count = len(lines)
    for item in lines:
        points = item[0]
        payload = item[1]
        text = str(payload[0]).strip() if payload else ""
        if not text or len(text) < MIN_TEXT_CHARS:
            short_filtered += 1
            continue
        confidence: Optional[float] = None
        if len(payload) > 1 and isinstance(payload[1], (int, float)):
            confidence = float(payload[1])
        bbox = _normalize_bbox(points)
        if bbox is None:
            continue
        blocks.append((bbox, text, confidence))
    capped = 0
    if MAX_TEXT_BLOCKS_PER_PAGE and len(blocks) > MAX_TEXT_BLOCKS_PER_PAGE:
        capped = len(blocks) - MAX_TEXT_BLOCKS_PER_PAGE
        blocks = blocks[:MAX_TEXT_BLOCKS_PER_PAGE]
    return blocks, raw_count, short_filtered, capped


def _ocr_summary(ocr_result: Any) -> str:
    parts: List[str] = [f"type={type(ocr_result).__name__}"]
    if isinstance(ocr_result, dict):
        parts.append(f"keys={list(ocr_result.keys())}")
    if isinstance(ocr_result, (list, tuple, dict)):
        try:
            parts.append(f"len={len(ocr_result)}")
        except Exception:
            pass
    preview_items: List[Any] = []
    if isinstance(ocr_result, list):
        preview_items = ocr_result[:2]
    elif isinstance(ocr_result, dict):
        preview_items = list(ocr_result.values())[:2]
    preview_chunks = []
    for item in preview_items:
        if isinstance(item, (list, tuple)):
            preview_chunks.append(
                f"{type(item).__name__}({len(item)})"
            )
        elif isinstance(item, dict):
            preview_chunks.append(f"dict(keys={list(item.keys())})")
        else:
            preview_chunks.append(type(item).__name__)
    if preview_chunks:
        parts.append(f"preview={preview_chunks}")
    return " ".join(parts)


def _json_safe_ocr_debug(ocr_result: Any) -> Dict[str, Any]:
    return {
        "type": type(ocr_result).__name__,
        "summary": _ocr_summary(ocr_result),
        "repr": repr(ocr_result)[:1000],
    }


def _build_parse_failed_record(
    page: int,
    timing_ms: int,
) -> Dict[str, Any]:
    record = _build_record(
        page=page,
        record_type="ocr_text",
        text="OCR_FALLBACK_PARSE_FAILED",
        bbox=None,
        timing_ms=timing_ms,
        block_type="text",
        confidence=None,
    )
    record["meta"]["parse_failed"] = True
    return record


def _count_block_types(result: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        block_type = _normalize_block_type(str(item.get("type", "")))
        counts[block_type] = counts.get(block_type, 0) + 1
    return counts


def _figure_fullpage_ratio(
    result: Sequence[Dict[str, Any]],
    image_path: Path,
) -> Optional[float]:
    figure_items = [
        item
        for item in result
        if isinstance(item, dict)
        and _normalize_block_type(str(item.get("type", ""))) == "figure"
    ]
    if len(figure_items) != 1:
        return None
    bbox = _normalize_bbox(figure_items[0].get("bbox"))
    if bbox is None:
        return None
    page_area = _page_area(image_path)
    if page_area is None or page_area <= 0:
        return None
    return _bbox_area(bbox) / page_area


def _should_run_ocr_fallback(
    result: Sequence[Dict[str, Any]],
    image_path: Path,
    force: bool = False,
) -> Tuple[bool, str, Dict[str, int]]:
    counts = _count_block_types(result)
    text_like_count = sum(
        counts.get(block, 0) for block in ("text", "title", "list", "equation")
    )
    table_count = counts.get("table", 0)
    no_text_or_table = text_like_count == 0 and table_count == 0

    if force:
        return True, "forced", counts
    if no_text_or_table:
        return True, "no_text_or_table", counts

    ratio = _figure_fullpage_ratio(result, image_path)
    if ratio is not None and ratio >= FULLPAGE_FIGURE_AREA_RATIO:
        return True, "fullpage_figure", counts

    return False, "has_text_or_table", counts


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


def _json_fallback(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return str(obj)
    return str(obj)


def run_paddleocr(
    img_dir: str,
    pages: Optional[Sequence[int]],
    lang: str,
    use_gpu: bool,
    cache_dir: Optional[str],
    debug_dir: Optional[str] = None,
    verbose: bool = False,
    force_ocr_fallback: bool = False,
) -> List[Dict[str, Any]]:
    cache_path = _configure_cache_dir(cache_dir)
    if cache_path is None:
        raise SystemExit(
            "PaddleOCR cache directory is not configured. "
            "Set BOOKMIND_PADDLEOCR_CACHE_DIR or run prefetch_online.sh first."
        )

    try:
        from paddleocr import PPStructure, PaddleOCR
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
    ocr_engine: Optional[Any] = None

    records: List[Dict[str, Any]] = []
    debug_path: Optional[Path] = None
    ocr_debug_path: Optional[Path] = None
    if debug_dir:
        debug_path = Path(debug_dir) / "paddleocr_raw"
        debug_path.mkdir(parents=True, exist_ok=True)
        ocr_debug_path = Path(debug_dir) / "paddleocr_ocr_raw"
        ocr_debug_path.mkdir(parents=True, exist_ok=True)
    for page_num, image_path in images:
        start = time.perf_counter()
        try:
            result = engine(str(image_path))
        except Exception as exc:
            raise SystemExit(f"PaddleOCR failed on {image_path}: {exc}")
        timing_ms = int(round((time.perf_counter() - start) * 1000))
        if debug_path is not None:
            raw_path = debug_path / f"page_{page_num}.json"
            raw_path.write_text(
                json.dumps(result, ensure_ascii=True, indent=2, default=_json_fallback)
            )

        page_records: List[Dict[str, Any]] = []
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
                page_records.append(
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
                page_records.append(
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
            page_records.append(
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

        run_fallback, reason, counts = _should_run_ocr_fallback(
            result,
            image_path,
            force=force_ocr_fallback,
        )
        if verbose:
            print(
                "PaddleOCR page "
                f"{page_num}: counts={counts} fallback={run_fallback} reason={reason}"
            )
        if run_fallback:
            if ocr_engine is None:
                ocr_engine = PaddleOCR(
                    show_log=False,
                    use_gpu=use_gpu,
                    lang=lang,
                    det=True,
                    rec=True,
                )
            ocr_start = time.perf_counter()
            try:
                ocr_result = ocr_engine.ocr(str(image_path), cls=False)
            except Exception as exc:
                raise SystemExit(f"PaddleOCR OCR fallback failed on {image_path}: {exc}")
            if verbose:
                print(f"PaddleOCR page {page_num}: ocr_raw_summary {_ocr_summary(ocr_result)}")
            ocr_timing_ms = int(round((time.perf_counter() - ocr_start) * 1000))
            page_timing_ms = timing_ms + ocr_timing_ms
            if ocr_debug_path is not None:
                raw_path = ocr_debug_path / f"page_{page_num}.json"
                try:
                    raw_path.write_text(
                        json.dumps(
                            ocr_result,
                            ensure_ascii=True,
                            indent=2,
                            default=_json_fallback,
                        )
                    )
                except Exception:
                    raw_path.write_text(
                        json.dumps(
                            _json_safe_ocr_debug(ocr_result),
                            ensure_ascii=True,
                            indent=2,
                        )
                    )
            for record in page_records:
                record["timing_ms"] = page_timing_ms
            (
                ocr_blocks,
                ocr_raw_count,
                ocr_short_filtered,
                ocr_capped,
            ) = _extract_ocr_text_blocks(ocr_result)
            if verbose:
                print(
                    "PaddleOCR page "
                    f"{page_num}: ocr_raw={ocr_raw_count} "
                    f"filtered_short={ocr_short_filtered} "
                    f"capped={ocr_capped} "
                    f"emitted={len(ocr_blocks)}"
                )
            if not ocr_blocks:
                page_records.append(_build_parse_failed_record(page_num, page_timing_ms))
            for bbox, text, confidence in ocr_blocks:
                page_records.append(
                    _build_record(
                        page=page_num,
                        record_type="ocr_text",
                        text=text,
                        bbox=bbox,
                        timing_ms=page_timing_ms,
                        block_type="text",
                        confidence=confidence,
                    )
                )

        records.extend(page_records)

    return records


def run_paddleocr_on_image(
    image_path: str,
    lang: str,
    use_gpu: bool,
    cache_dir: Optional[str],
    debug_dir: Optional[str] = None,
    verbose: bool = False,
    force_ocr_fallback: bool = False,
) -> List[Dict[str, Any]]:
    image_path = str(image_path)
    cache_path = _configure_cache_dir(cache_dir)
    if cache_path is None:
        raise SystemExit(
            "PaddleOCR cache directory is not configured. "
            "Set BOOKMIND_PADDLEOCR_CACHE_DIR or run prefetch_online.sh first."
        )

    try:
        from paddleocr import PPStructure, PaddleOCR
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

    ocr_debug_path: Optional[Path] = None
    if debug_dir:
        ocr_debug_path = Path(debug_dir) / "paddleocr_ocr_raw"
        ocr_debug_path.mkdir(parents=True, exist_ok=True)
    run_fallback, reason, counts = _should_run_ocr_fallback(
        result,
        Path(image_path),
        force=force_ocr_fallback,
    )
    if verbose:
        print(
            "PaddleOCR page 1: counts="
            f"{counts} fallback={run_fallback} reason={reason}"
        )
    if run_fallback:
        ocr_engine = PaddleOCR(
            show_log=False,
            use_gpu=use_gpu,
            lang=lang,
            det=True,
            rec=True,
        )
        ocr_start = time.perf_counter()
        try:
            ocr_result = ocr_engine.ocr(image_path, cls=False)
        except Exception as exc:
            raise SystemExit(f"PaddleOCR OCR fallback failed on {image_path}: {exc}")
        if verbose:
            print(f"PaddleOCR page 1: ocr_raw_summary {_ocr_summary(ocr_result)}")
        ocr_timing_ms = int(round((time.perf_counter() - ocr_start) * 1000))
        page_timing_ms = timing_ms + ocr_timing_ms
        if ocr_debug_path is not None:
            raw_path = ocr_debug_path / "page_1.json"
            try:
                raw_path.write_text(
                    json.dumps(
                        ocr_result,
                        ensure_ascii=True,
                        indent=2,
                        default=_json_fallback,
                    )
                )
            except Exception:
                raw_path.write_text(
                    json.dumps(
                        _json_safe_ocr_debug(ocr_result),
                        ensure_ascii=True,
                        indent=2,
                    )
                )
        for record in records:
            record["timing_ms"] = page_timing_ms
        (
            ocr_blocks,
            ocr_raw_count,
            ocr_short_filtered,
            ocr_capped,
        ) = _extract_ocr_text_blocks(ocr_result)
        if verbose:
            print(
                "PaddleOCR page 1: "
                f"ocr_raw={ocr_raw_count} "
                f"filtered_short={ocr_short_filtered} "
                f"capped={ocr_capped} "
                f"emitted={len(ocr_blocks)}"
            )
        if not ocr_blocks:
            records.append(_build_parse_failed_record(1, page_timing_ms))
        for bbox, text, confidence in ocr_blocks:
            records.append(
                _build_record(
                    page=1,
                    record_type="ocr_text",
                    text=text,
                    bbox=bbox,
                    timing_ms=page_timing_ms,
                    block_type="text",
                    confidence=confidence,
                )
            )

    return records
