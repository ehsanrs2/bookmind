"""Merge PaddleOCR and VLM JSONL outputs into ingestion-ready records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class MergeStats:
    output_path: Path
    total_emitted: int
    emitted_by_type: Dict[str, int]
    figure_captions_matched: int


def _read_jsonl_with_line_ids(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_id, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield line_id, row


def _collapse_ws(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip())


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return _collapse_ws(str(value))


def _normalize_bbox(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in value):
        return None
    x0, y0, x1, y1 = (float(v) for v in value)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _bbox_to_ints(bbox: Optional[Sequence[float]]) -> Optional[List[int]]:
    if bbox is None:
        return None
    return [int(round(float(v))) for v in bbox]


def _bbox_norm_token(bbox: Optional[Sequence[float]]) -> str:
    ints = _bbox_to_ints(bbox)
    if ints is None:
        return "null"
    return ",".join(str(v) for v in ints)


def _stable_id(page: int, content_type: str, bbox: Optional[Sequence[float]], text: str) -> str:
    text_norm = _norm_text(text)
    token = f"{page}|{content_type}|{_bbox_norm_token(bbox)}|{text_norm}"
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]


def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _is_exact_match(box_a: Sequence[float], box_b: Sequence[float]) -> bool:
    return all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(box_a, box_b))


def _is_within_tol(box_a: Sequence[float], box_b: Sequence[float], bbox_tol: float) -> bool:
    return all(abs(float(a) - float(b)) <= bbox_tol for a, b in zip(box_a, box_b))


def _pick_caption_for_figure(
    figure_bbox: Sequence[float],
    candidates: Iterable[Dict[str, Any]],
    bbox_tol: float,
    iou_threshold: float,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score: Optional[Tuple[int, float, float]] = None
    for candidate in candidates:
        cbbox = candidate["bbox"]
        priority = -1
        quality = 0.0
        if _is_exact_match(figure_bbox, cbbox):
            priority = 3
            quality = 1.0
        elif _is_within_tol(figure_bbox, cbbox, bbox_tol):
            priority = 2
            max_delta = max(abs(a - b) for a, b in zip(figure_bbox, cbbox))
            quality = 1.0 / (1.0 + max_delta)
        else:
            iou = _bbox_iou(figure_bbox, cbbox)
            if iou >= iou_threshold:
                priority = 1
                quality = iou
        if priority < 0:
            continue
        # Keep deterministic ordering with line_id as final tiebreaker.
        score = (priority, quality, -float(candidate["line_id"]))
        if best_score is None or score > best_score:
            best_score = score
            best = candidate
    return best


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def _build_ingest_record(
    *,
    page: int,
    content_type: str,
    text: str,
    bbox: Optional[Sequence[float]],
    source_engines: List[str],
    block_type: Optional[str],
    confidence: Optional[float],
    paddle_line_ids: List[int],
    vlm_line_id: Optional[int],
) -> Dict[str, Any]:
    text_norm = _norm_text(text)
    bbox_norm = _bbox_to_ints(bbox)
    return {
        "stable_id": _stable_id(page=page, content_type=content_type, bbox=bbox, text=text_norm),
        "page": page,
        "content_type": content_type,
        "text": text_norm,
        "bbox": bbox_norm,
        "meta": {
            "pdf_page_start": page,
            "pdf_page_end": page,
            "source_engines": source_engines,
            "block_type": block_type,
            "confidence": confidence,
            "trace": {
                "paddleocr_line_ids": paddle_line_ids,
                "vlm_line_id": vlm_line_id,
            },
        },
    }


def merge_outputs(
    *,
    paddle_jsonl: Path,
    out_jsonl: Path,
    vlm_jsonl: Optional[Path] = None,
    min_text_chars: int = 20,
    bbox_tol: int = 2,
    iou_threshold: float = 0.95,
) -> MergeStats:
    if not paddle_jsonl.exists():
        raise SystemExit(f"PaddleOCR JSONL not found: {paddle_jsonl}")
    if vlm_jsonl is not None and not vlm_jsonl.exists():
        raise SystemExit(f"VLM JSONL not found: {vlm_jsonl}")

    vlm_by_page: Dict[int, List[Dict[str, Any]]] = {}
    if vlm_jsonl is not None:
        for line_id, row in _read_jsonl_with_line_ids(vlm_jsonl):
            if row.get("type") != "figure_caption":
                continue
            page = row.get("page")
            bbox = _normalize_bbox(row.get("bbox"))
            text = _norm_text(row.get("text"))
            if not isinstance(page, int) or bbox is None or not text:
                continue
            vlm_by_page.setdefault(page, []).append(
                {
                    "line_id": line_id,
                    "bbox": bbox,
                    "text": text,
                    "confidence": row.get("meta", {}).get("confidence")
                    if isinstance(row.get("meta"), dict)
                    else None,
                }
            )

    out_rows: List[Dict[str, Any]] = []
    emitted_by_type: Dict[str, int] = {"text": 0, "table": 0, "figure_caption": 0}
    matched_captions = 0

    for line_id, row in _read_jsonl_with_line_ids(paddle_jsonl):
        page = row.get("page")
        record_type = row.get("type")
        text = _norm_text(row.get("text"))
        bbox = _normalize_bbox(row.get("bbox"))
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        block_type = meta.get("block_type")
        confidence = meta.get("confidence")

        if not isinstance(page, int):
            continue

        if record_type == "ocr_text":
            if len(text) < min_text_chars:
                continue
            record = _build_ingest_record(
                page=page,
                content_type="text",
                text=text,
                bbox=bbox,
                source_engines=["paddleocr"],
                block_type=block_type or "text",
                confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                paddle_line_ids=[line_id],
                vlm_line_id=None,
            )
            out_rows.append(record)
            emitted_by_type["text"] += 1
            continue

        if record_type == "table_md":
            if not text:
                continue
            record = _build_ingest_record(
                page=page,
                content_type="table",
                text=text,
                bbox=bbox,
                source_engines=["paddleocr"],
                block_type=block_type or "table",
                confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                paddle_line_ids=[line_id],
                vlm_line_id=None,
            )
            out_rows.append(record)
            emitted_by_type["table"] += 1
            continue

        if record_type == "layout_block" and str(block_type).lower() == "figure" and bbox is not None:
            candidates = vlm_by_page.get(page, [])
            if not candidates:
                continue
            match = _pick_caption_for_figure(
                figure_bbox=bbox,
                candidates=candidates,
                bbox_tol=float(bbox_tol),
                iou_threshold=iou_threshold,
            )
            if match is None:
                continue
            candidates.remove(match)
            record = _build_ingest_record(
                page=page,
                content_type="figure_caption",
                text=match["text"],
                bbox=bbox,
                source_engines=["paddleocr", "vlm"],
                block_type="figure",
                confidence=float(match["confidence"])
                if isinstance(match.get("confidence"), (int, float))
                else (float(confidence) if isinstance(confidence, (int, float)) else None),
                paddle_line_ids=[line_id],
                vlm_line_id=int(match["line_id"]),
            )
            out_rows.append(record)
            emitted_by_type["figure_caption"] += 1
            matched_captions += 1

    _append_jsonl(out_jsonl, out_rows)
    return MergeStats(
        output_path=out_jsonl,
        total_emitted=len(out_rows),
        emitted_by_type=emitted_by_type,
        figure_captions_matched=matched_captions,
    )
