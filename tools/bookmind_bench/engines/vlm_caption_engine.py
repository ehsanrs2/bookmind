"""Offline VLM caption runner for figure crops."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from engines.crop_utils import crop_image
from engines.layout_engine import list_page_images
from PIL import Image


DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "qwen3-vl"
DEFAULT_PROMPT_TEMPLATE = "technical_diagram_v1"
DEFAULT_IOU_THRESHOLD = 0.01
DEFAULT_OCR_MAX_CHARS = 800
DEFAULT_OCR_MAX_ITEMS = 30
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2


@dataclass
class OcrBlock:
    bbox: List[float]
    text: str
    region_id: Optional[str] = None


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_jsonl_with_line_ids(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_id, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield line_id, row


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")


def _read_jobs(path: Path) -> Iterator[Dict[str, Any]]:
    yield from _read_jsonl(path)


def _normalize_bbox(bbox: Iterable[float]) -> List[float]:
    values = list(bbox)
    if len(values) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(values)}")
    return values


def _bbox_iou(bbox_a: Iterable[float], bbox_b: Iterable[float]) -> float:
    a = _normalize_bbox(bbox_a)
    b = _normalize_bbox(bbox_b)
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    left = max(min(ax0, ax1), min(bx0, bx1))
    top = max(min(ay0, ay1), min(by0, by1))
    right = min(max(ax0, ax1), max(bx0, bx1))
    bottom = min(max(ay0, ay1), max(by0, by1))

    inter_w = max(0.0, right - left)
    inter_h = max(0.0, bottom - top)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = abs((ax1 - ax0) * (ay1 - ay0))
    area_b = abs((bx1 - bx0) * (by1 - by0))
    if area_a <= 0 or area_b <= 0:
        return 0.0

    return inter_area / (area_a + area_b - inter_area)


def _load_ocr_blocks(paddleocr_jsonl: Optional[Path]) -> Dict[int, List[OcrBlock]]:
    if paddleocr_jsonl is None:
        return {}

    ocr_by_page: Dict[int, List[OcrBlock]] = {}
    for row in _read_jsonl(paddleocr_jsonl):
        if row.get("type") != "ocr_text":
            continue
        page = row.get("page")
        bbox = row.get("bbox")
        text = row.get("text")
        if not isinstance(page, int):
            continue
        if not isinstance(bbox, list) or text is None:
            continue
        ocr_by_page.setdefault(page, []).append(
            OcrBlock(bbox=[float(v) for v in bbox], text=str(text), region_id=None)
        )
    return ocr_by_page


def _extract_region_id(row: Dict[str, Any]) -> Optional[str]:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return None
    trace = meta.get("trace")
    if isinstance(trace, dict):
        rid = trace.get("region_id")
        if rid is not None:
            return str(rid)
    rid = meta.get("region_id")
    if rid is None:
        return None
    return str(rid)


def _load_layout_ocr_blocks(layout_jsonl: Path) -> Dict[int, List[OcrBlock]]:
    ocr_by_page: Dict[int, List[OcrBlock]] = {}
    for _, row in _read_jsonl_with_line_ids(layout_jsonl):
        if row.get("type") != "ocr_text":
            continue
        page = row.get("page")
        bbox = row.get("bbox")
        text = row.get("text")
        if not isinstance(page, int):
            continue
        if not isinstance(bbox, list) or text is None:
            continue
        ocr_by_page.setdefault(page, []).append(
            OcrBlock(
                bbox=[float(v) for v in bbox],
                text=str(text),
                region_id=_extract_region_id(row),
            )
        )
    return ocr_by_page


def _select_nearby_page_hints(
    figure_bbox: Iterable[float],
    blocks: Iterable[OcrBlock],
    max_chars: int = DEFAULT_OCR_MAX_CHARS,
) -> str:
    fbox = _normalize_bbox(figure_bbox)
    fx = (fbox[0] + fbox[2]) / 2.0
    fy = (fbox[1] + fbox[3]) / 2.0
    scored: List[Tuple[float, str]] = []
    for block in blocks:
        text = " ".join(block.text.strip().split())
        if not text:
            continue
        b = _normalize_bbox(block.bbox)
        bx = (b[0] + b[2]) / 2.0
        by = (b[1] + b[3]) / 2.0
        dist2 = (bx - fx) ** 2 + (by - fy) ** 2
        scored.append((dist2, text))

    if not scored:
        return ""

    scored.sort(key=lambda item: item[0])
    chunks: List[str] = []
    used = 0
    for _, text in scored:
        sep = 2 if chunks else 0
        if used + sep + len(text) > max_chars:
            break
        if sep:
            used += sep
        chunks.append(text)
        used += len(text)
    return "; ".join(chunks)


def _join_hint_texts(blocks: Iterable[OcrBlock], max_chars: int) -> str:
    chunks: List[str] = []
    used = 0
    for block in blocks:
        text = " ".join(block.text.strip().split())
        if not text:
            continue
        sep = 2 if chunks else 0
        if used + sep + len(text) > max_chars:
            break
        if sep:
            used += sep
        chunks.append(text)
        used += len(text)
    return "; ".join(chunks)


def _select_ocr_hints(
    figure_bbox: Iterable[float],
    blocks: Iterable[OcrBlock],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_chars: int = DEFAULT_OCR_MAX_CHARS,
    max_items: int = DEFAULT_OCR_MAX_ITEMS,
) -> Tuple[str, bool]:
    selected: List[str] = []
    for block in blocks:
        if _bbox_iou(figure_bbox, block.bbox) < iou_threshold:
            continue
        text = " ".join(block.text.strip().split())
        if not text:
            continue
        selected.append(text)
        if len(selected) >= max_items:
            break

    if not selected:
        return "", False

    joined = "; ".join(selected)
    if len(joined) > max_chars:
        joined = joined[: max_chars - 3].rstrip() + "..."
    return joined, True


def _build_prompt(prompt_template: str, ocr_hints: str) -> str:
    if prompt_template != DEFAULT_PROMPT_TEMPLATE:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    base_prompt = (
        "You are captioning a technical diagram or figure. "
        "Describe the components and labels, connections or flow between them, "
        "the inputs/outputs, the purpose of each block, and any warnings or notes. "
        "Be concise but specific."
    )
    if not ocr_hints:
        return base_prompt

    return f"{base_prompt}\n\nOCR hints: {ocr_hints}"


def _encode_image_base64(image_path: Path) -> str:
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _post_chat_completion(endpoint: str, payload: Dict[str, Any], timeout_s: int = 60) -> Dict[str, Any]:
    url = endpoint.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_caption(response: Dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("VLM response missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("VLM response missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("VLM response missing content")
    return content.strip()


def _build_payload(
    model: str,
    prompt: str,
    image_path: Path,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    image_b64 = _encode_image_base64(image_path)
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _resize_crop(path: Path, max_side: int) -> None:
    if max_side <= 0:
        return
    with Image.open(path) as image:
        width, height = image.size
        longest = max(width, height)
        if longest <= max_side:
            return
        scale = float(max_side) / float(longest)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized = image.resize(new_size)
        resized.save(path, format="PNG")


def _resolve_page_image_map(imgdir: Path) -> Dict[int, Path]:
    return {page: path for page, path in list_page_images(str(imgdir))}


def run_vlm_layout(
    *,
    layout_jsonl: str,
    imgdir: str,
    out_dir: str,
    endpoint: str = DEFAULT_ENDPOINT,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    ocr_hint_max_chars: int = DEFAULT_OCR_MAX_CHARS,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    limit_image_size: Optional[int] = None,
) -> int:
    layout_path = Path(layout_jsonl)
    imgdir_path = Path(imgdir)
    out_dir_path = Path(out_dir)
    if not layout_path.exists():
        raise SystemExit(f"Layout JSONL not found: {layout_path}")
    if not imgdir_path.exists():
        raise SystemExit(f"Image directory not found: {imgdir_path}")

    if model is None:
        model = os.environ.get("BOOKMIND_VLM_MODEL", DEFAULT_MODEL)

    image_map = _resolve_page_image_map(imgdir_path)
    ocr_index = _load_layout_ocr_blocks(layout_path)
    out_path = out_dir_path / "vlm" / "output.jsonl"

    written = 0
    for line_id, row in _read_jsonl_with_line_ids(layout_path):
        if row.get("type") != "layout_block":
            continue
        page = row.get("page")
        bbox = row.get("bbox")
        meta = row.get("meta")
        if not isinstance(meta, dict):
            continue
        if str(meta.get("block_type", "")).lower() != "figure":
            continue
        if not isinstance(page, int) or not isinstance(bbox, list):
            continue
        image_path = image_map.get(page)
        if image_path is None:
            continue

        region_id = _extract_region_id(row)
        crop_path = out_dir_path / "vlm" / "crops" / f"page_{page:04d}_line_{line_id:05d}.png"
        try:
            crop_image(str(image_path), bbox, str(crop_path), pad_px=12)
            if limit_image_size is not None:
                _resize_crop(crop_path, int(limit_image_size))
        except Exception as exc:  # noqa: BLE001
            print(f"[vlm-layout] Warning: failed to crop line {line_id} ({exc}); skipping")
            continue

        page_blocks = ocr_index.get(page, [])
        region_blocks = (
            [block for block in page_blocks if region_id and block.region_id == region_id]
            if region_id
            else []
        )
        ocr_hints = ""
        used_region_hints = False
        if region_blocks:
            ocr_hints = _join_hint_texts(region_blocks, ocr_hint_max_chars)
            used_region_hints = bool(ocr_hints)
        if not ocr_hints:
            ocr_hints = _select_nearby_page_hints(
                bbox,
                page_blocks,
                max_chars=ocr_hint_max_chars,
            )

        prompt = _build_prompt(prompt_template, ocr_hints)
        start = time.perf_counter()
        try:
            payload = _build_payload(
                model,
                prompt,
                crop_path,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = _post_chat_completion(endpoint, payload)
            caption = _extract_caption(response)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                body = ""
            print(
                f"[vlm-layout] Warning: line {line_id} failed (HTTP {exc.code}: {body}); skipping"
            )
            continue
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"VLM endpoint unreachable at {endpoint}. Start vLLM with OpenAI-compatible API."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            print(f"[vlm-layout] Warning: line {line_id} failed ({exc}); skipping")
            continue
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))

        record = {
            "engine": "vlm",
            "page": page,
            "type": "figure_caption",
            "text": caption,
            "bbox": bbox,
            "timing_ms": elapsed_ms,
            "meta": {
                "pdf_page_start": row.get("meta", {}).get("pdf_page_start", page),
                "pdf_page_end": row.get("meta", {}).get("pdf_page_end", page),
                "prompt_template": prompt_template,
                "model": model,
                "job_index": written + 1,
                "used_ocr_hints": bool(ocr_hints),
                "used_region_hints": used_region_hints,
                "trace": {
                    "layout_region_id": region_id,
                    "layout_line_id": line_id,
                },
            },
        }
        _append_jsonl(out_path, record)
        written += 1

    return written


def run_vlm_caption_jobs(
    jobs_jsonl: str,
    out_dir: str,
    endpoint: str = DEFAULT_ENDPOINT,
    model: Optional[str] = None,
    paddleocr_jsonl: Optional[str] = None,
    use_ocr_hints: bool = True,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> int:
    jobs_path = Path(jobs_jsonl)
    if not jobs_path.exists():
        raise SystemExit(f"Jobs JSONL not found: {jobs_path}")

    out_dir_path = Path(out_dir)
    out_path = out_dir_path / "vlm" / "output.jsonl"

    if model is None:
        model = os.environ.get("BOOKMIND_VLM_MODEL", DEFAULT_MODEL)

    ocr_index = _load_ocr_blocks(Path(paddleocr_jsonl)) if paddleocr_jsonl else {}

    written = 0
    for idx, job in enumerate(_read_jobs(jobs_path), start=1):
        page = job.get("page")
        bbox = job.get("bbox")
        image_path = job.get("image_path")
        crop_path = job.get("suggested_crop_path")
        if not isinstance(page, int) or not isinstance(bbox, list) or not image_path or not crop_path:
            print(f"[vlm] Skipping malformed job at index {idx}")
            continue

        try:
            crop_image(image_path, bbox, crop_path, pad_px=12)
        except Exception as exc:  # noqa: BLE001
            print(f"[vlm] Warning: failed to crop job {idx} ({exc}); skipping")
            continue

        ocr_hints = ""
        used_hints = False
        if use_ocr_hints and ocr_index:
            blocks = ocr_index.get(page, [])
            ocr_hints, used_hints = _select_ocr_hints(bbox, blocks)

        prompt_template = job.get("prompt_template") or DEFAULT_PROMPT_TEMPLATE
        prompt = _build_prompt(str(prompt_template), ocr_hints)

        start = time.perf_counter()
        try:
            payload = _build_payload(
                model,
                prompt,
                Path(crop_path),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = _post_chat_completion(endpoint, payload)
            caption = _extract_caption(response)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                body = ""
            print(
                f"[vlm] Warning: VLM job {idx} failed (HTTP {exc.code}: {body}); skipping"
            )
            continue
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"VLM endpoint unreachable at {endpoint}. Start vLLM with OpenAI-compatible API."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            print(f"[vlm] Warning: VLM job {idx} failed ({exc}); skipping")
            continue
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))

        record = {
            "engine": "vlm",
            "page": page,
            "type": "figure_caption",
            "text": caption,
            "bbox": bbox,
            "timing_ms": elapsed_ms,
            "meta": {
                "pdf_page_start": job.get("pdf_page_start", page),
                "pdf_page_end": job.get("pdf_page_end", page),
                "prompt_template": prompt_template,
                "model": model,
                "job_index": idx,
                "used_ocr_hints": used_hints,
            },
        }
        _append_jsonl(out_path, record)
        written += 1

    return written
