"""LayoutParser + Detectron2 region-aware OCR engine for Bookmind bench."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

_PAGE_RE = re.compile(r"(?:page|pg|p)[_\- ]*(\d+)", re.IGNORECASE)

_PUBLAYNET_LABEL_MAP = {
    0: "text",
    1: "title",
    2: "list",
    3: "table",
    4: "figure",
}


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

    ordered = sorted(img_path.glob("*.png"))
    if not ordered:
        return []

    images: List[Tuple[int, Path]] = []
    fallback_page = 1
    for path in ordered:
        page_num = _page_num_from_name(path)
        if page_num is None:
            page_num = fallback_page
        fallback_page += 1
        images.append((page_num, path))

    images.sort(key=lambda item: (item[0], item[1].name))
    return images


def clamp_bbox_to_image(
    bbox: Iterable[float],
    image_width: int,
    image_height: int,
) -> Optional[List[float]]:
    values = [float(v) for v in bbox]
    if len(values) != 4:
        raise ValueError("bbox must contain 4 values")

    x0, y0, x1, y1 = values
    left = max(0.0, min(float(image_width), min(x0, x1)))
    right = max(0.0, min(float(image_width), max(x0, x1)))
    top = max(0.0, min(float(image_height), min(y0, y1)))
    bottom = max(0.0, min(float(image_height), max(y0, y1)))

    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def map_crop_bbox_to_page(
    crop_bbox: Iterable[float],
    region_bbox: Sequence[float],
    page_width: int,
    page_height: int,
) -> Optional[List[float]]:
    crop_values = [float(v) for v in crop_bbox]
    if len(crop_values) != 4:
        raise ValueError("crop_bbox must contain 4 values")
    if len(region_bbox) != 4:
        raise ValueError("region_bbox must contain 4 values")

    rx0, ry0, _, _ = [float(v) for v in region_bbox]
    x0, y0, x1, y1 = crop_values
    page_bbox = [rx0 + x0, ry0 + y0, rx0 + x1, ry0 + y1]
    return clamp_bbox_to_image(page_bbox, page_width, page_height)


def _normalize_bbox(bbox: Any) -> Optional[List[float]]:
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)):
        if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            return [float(v) for v in bbox]
        if len(bbox) == 4 and all(
            isinstance(v, (list, tuple)) and len(v) == 2 for v in bbox
        ):
            xs = [float(v[0]) for v in bbox]
            ys = [float(v[1]) for v in bbox]
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _extract_element_bbox(element: Any) -> Optional[List[float]]:
    for candidate in (
        getattr(element, "coordinates", None),
        getattr(element, "bbox", None),
    ):
        normalized = _normalize_bbox(candidate)
        if normalized is not None:
            return normalized

    for container in (element, getattr(element, "block", None)):
        if container is None:
            continue
        coords = [
            getattr(container, "x_1", None),
            getattr(container, "y_1", None),
            getattr(container, "x_2", None),
            getattr(container, "y_2", None),
        ]
        if all(isinstance(v, (int, float)) for v in coords):
            return [float(v) for v in coords]

    return None


def _normalize_block_type(raw_type: str) -> Optional[str]:
    block_type = raw_type.strip().lower()
    if block_type in {"text", "title", "list"}:
        return "text"
    if block_type in {"table", "figure"}:
        return block_type
    return None


def _extract_confidence(element: Any) -> Optional[float]:
    for value in (
        getattr(element, "score", None),
        getattr(getattr(element, "block", None), "score", None),
    ):
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_raw_type(element: Any) -> str:
    for value in (
        getattr(element, "type", None),
        getattr(getattr(element, "block", None), "type", None),
    ):
        if value is not None:
            return str(value)
    return "unknown"


def _find_model_file(model_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        path = model_dir / candidate
        if path.exists() and path.is_file():
            return path
    return None


def _find_first_with_suffix(model_dir: Path, suffixes: Sequence[str]) -> Optional[Path]:
    for suffix in suffixes:
        found = sorted(model_dir.rglob(f"*{suffix}"))
        if found:
            return found[0]
    return None


def _resolve_layout_model_paths(model_dir: str) -> Tuple[Path, Path]:
    root = Path(model_dir)
    if not root.exists():
        raise SystemExit(
            f"Layout model directory not found: {root}. "
            "Run tools/bookmind_bench/scripts/prefetch_online.sh --profile layout first."
        )

    config = _find_model_file(
        root,
        (
            "config.yaml",
            "config.yml",
            "publaynet/config.yaml",
            "publaynet/config.yml",
        ),
    ) or _find_first_with_suffix(root, (".yaml", ".yml"))

    weights = _find_model_file(
        root,
        (
            "model_final.pth",
            "model.pth",
            "publaynet/model_final.pth",
            "publaynet/model.pth",
        ),
    ) or _find_first_with_suffix(root, (".pth", ".pkl"))

    if config is None or weights is None:
        raise SystemExit(
            "LayoutParser assets missing. Expected a local Detectron2 config (.yaml) "
            f"and weights (.pth/.pkl) under {root}."
        )

    return config, weights


def _configure_paddle_cache() -> Path:
    override = os.environ.get("BOOKMIND_PADDLEOCR_CACHE_DIR")
    default_dir = (
        Path(__file__).resolve().parents[1]
        / "offline_bundle"
        / "models"
        / "paddleocr"
    )
    cache_dir = Path(override) if override else default_dir

    if not cache_dir.exists() or not any(cache_dir.iterdir()):
        raise SystemExit(
            f"PaddleOCR cache directory not found or empty: {cache_dir}. "
            "Run tools/bookmind_bench/scripts/prefetch_online.sh --profile ocr first."
        )

    os.environ["PADDLEOCR_HOME"] = str(cache_dir)
    os.environ["PADDLE_HOME"] = str(cache_dir)
    os.environ["HOME"] = str(cache_dir)
    return cache_dir


def _load_layout_model(model_dir: str, use_gpu: bool, score_thresh: float) -> Any:
    try:
        import layoutparser as lp
    except Exception as exc:
        raise SystemExit(f"Unable to import layoutparser: {exc}")

    config_path, model_path = _resolve_layout_model_paths(model_dir)
    device = "cuda" if use_gpu else "cpu"

    try:
        model = lp.Detectron2LayoutModel(
            config_path=str(config_path),
            model_path=str(model_path),
            label_map=_PUBLAYNET_LABEL_MAP,
            extra_config=[
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                float(score_thresh),
                "MODEL.DEVICE",
                device,
            ],
        )
    except Exception as exc:
        raise SystemExit(f"Failed to initialize LayoutParser Detectron2 model: {exc}")

    return model


def _load_ocr_engine(use_gpu: bool) -> Any:
    _configure_paddle_cache()
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise SystemExit(f"Unable to import PaddleOCR: {exc}")

    try:
        return PaddleOCR(show_log=False, use_gpu=use_gpu, det=True, rec=True, lang="en")
    except Exception as exc:
        raise SystemExit(f"Failed to initialize PaddleOCR OCR engine: {exc}")


def _normalize_ocr_lines(ocr_result: Any) -> List[Any]:
    if isinstance(ocr_result, list):
        if ocr_result and isinstance(ocr_result[0], list):
            if len(ocr_result) == 1 and ocr_result[0] and isinstance(ocr_result[0][0], (list, tuple)):
                return ocr_result[0]
            return ocr_result
    if isinstance(ocr_result, dict):
        if "result" in ocr_result and isinstance(ocr_result["result"], list):
            return ocr_result["result"]
    return []


def _extract_ocr_lines(
    ocr_result: Any,
) -> List[Tuple[List[float], str, Optional[float]]]:
    lines = _normalize_ocr_lines(ocr_result)
    out: List[Tuple[List[float], str, Optional[float]]] = []
    for item in lines:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        bbox = _normalize_bbox(item[0])
        if bbox is None:
            continue
        payload = item[1]
        text = ""
        confidence: Optional[float] = None
        if isinstance(payload, (list, tuple)) and payload:
            text = str(payload[0]).strip()
            if len(payload) > 1 and isinstance(payload[1], (int, float)):
                confidence = float(payload[1])
        elif isinstance(payload, str):
            text = payload.strip()
        if not text:
            continue
        out.append((bbox, text, confidence))
    return out


def _build_record(
    page: int,
    record_type: str,
    text: str,
    bbox: Optional[List[float]],
    timing_ms: int,
    block_type: Optional[str] = None,
    confidence: Optional[float] = None,
    trace_region_id: Optional[str] = None,
    region_id: Optional[str] = None,
    raw_block_type: Optional[str] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "pdf_page_start": page,
        "pdf_page_end": page,
    }
    if block_type:
        meta["block_type"] = block_type
    if confidence is not None:
        meta["confidence"] = confidence
    if region_id:
        meta["region_id"] = region_id
    if raw_block_type:
        meta["raw_block_type"] = raw_block_type
    if trace_region_id:
        meta["trace"] = {"region_id": trace_region_id}

    record: Dict[str, Any] = {
        "engine": "layoutparser",
        "page": page,
        "type": record_type,
        "text": text,
        "timing_ms": timing_ms,
        "meta": meta,
    }
    if bbox is not None:
        record["bbox"] = bbox
    return record


def validate_layout_record(record: Dict[str, Any]) -> None:
    if record.get("engine") != "layoutparser":
        raise ValueError("engine must be 'layoutparser'")
    if not isinstance(record.get("page"), int) or record["page"] < 1:
        raise ValueError("page must be a positive int")
    record_type = record.get("type")
    if record_type not in {"layout_block", "ocr_text", "table_md"}:
        raise ValueError("type must be layout_block, ocr_text, or table_md")
    if "text" not in record:
        raise ValueError("text is required")
    meta = record.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dict")
    for key in ("pdf_page_start", "pdf_page_end"):
        if key not in meta:
            raise ValueError(f"meta.{key} is required")
    if record_type == "layout_block" and not meta.get("block_type"):
        raise ValueError("layout_block records must include meta.block_type")
    if record_type == "ocr_text":
        trace = meta.get("trace")
        if not isinstance(trace, dict) or not trace.get("region_id"):
            raise ValueError("ocr_text records must include meta.trace.region_id")
    if "bbox" in record and record["bbox"] is not None:
        bbox = record["bbox"]
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)
        ):
            raise ValueError("bbox must be [x0, y0, x1, y1]")


def run_layoutparser(
    img_dir: str,
    pages: Optional[Sequence[int]],
    use_gpu: bool,
    model_dir: str,
    score_thresh: float,
    max_regions_per_page: int,
    out_dir: str,
) -> List[Dict[str, Any]]:
    images = list_page_images(img_dir)
    if not images:
        raise SystemExit(f"No page images found in {img_dir}")

    if pages:
        wanted = set(pages)
        images = [item for item in images if item[0] in wanted]
    if not images:
        raise SystemExit("No matching pages found in image directory.")

    layout_model = _load_layout_model(model_dir=model_dir, use_gpu=use_gpu, score_thresh=score_thresh)
    ocr_engine = _load_ocr_engine(use_gpu=use_gpu)

    layout_root = Path(out_dir) / "layout"
    crops_dir = layout_root / "crops"
    debug_dir = layout_root / "layout_debug"
    crops_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    for page_num, image_path in images:
        page_start = time.perf_counter()
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            page_w, page_h = rgb.size
            page_np = np.array(rgb)

        try:
            detected = layout_model.detect(page_np)
        except Exception as exc:
            raise SystemExit(f"LayoutParser detect failed on {image_path}: {exc}")

        region_candidates: List[Dict[str, Any]] = []
        for element in list(detected):
            raw_type = _extract_raw_type(element)
            block_type = _normalize_block_type(raw_type)
            if block_type is None:
                continue
            raw_bbox = _extract_element_bbox(element)
            if raw_bbox is None:
                continue
            bbox = clamp_bbox_to_image(raw_bbox, page_w, page_h)
            if bbox is None:
                continue
            region_candidates.append(
                {
                    "raw_block_type": raw_type,
                    "block_type": block_type,
                    "bbox": bbox,
                    "confidence": _extract_confidence(element),
                }
            )

        region_candidates.sort(key=lambda r: (r["bbox"][1], r["bbox"][0], r["bbox"][3], r["bbox"][2]))
        if max_regions_per_page > 0:
            region_candidates = region_candidates[:max_regions_per_page]

        page_layout_records: List[Dict[str, Any]] = []
        page_ocr_records: List[Dict[str, Any]] = []
        debug_payload: List[Dict[str, Any]] = []

        with Image.open(image_path) as source_image:
            source_rgb = source_image.convert("RGB")
            for idx, region in enumerate(region_candidates, start=1):
                bbox = region["bbox"]
                left, top, right, bottom = [int(round(v)) for v in bbox]
                if right <= left or bottom <= top:
                    continue

                region_id = f"page_{page_num:04d}_region_{idx:03d}"

                page_layout_records.append(
                    _build_record(
                        page=page_num,
                        record_type="layout_block",
                        text="",
                        bbox=bbox,
                        timing_ms=0,
                        block_type=region["block_type"],
                        confidence=region["confidence"],
                        region_id=region_id,
                        raw_block_type=region["raw_block_type"],
                    )
                )

                debug_payload.append(
                    {
                        "region_id": region_id,
                        "bbox": bbox,
                        "block_type": region["block_type"],
                        "raw_block_type": region["raw_block_type"],
                        "confidence": region["confidence"],
                    }
                )

                if region["block_type"] not in {"text", "table"}:
                    continue

                crop = source_rgb.crop((left, top, right, bottom))
                crop_path = crops_dir / f"page_{page_num:04d}_region_{idx:03d}.png"
                crop.save(crop_path, format="PNG")

                crop_np = np.array(crop)
                try:
                    ocr_result = ocr_engine.ocr(crop_np, cls=False)
                except Exception as exc:
                    raise SystemExit(f"PaddleOCR failed on region crop {crop_path}: {exc}")

                for line_bbox, text, confidence in _extract_ocr_lines(ocr_result):
                    page_bbox = map_crop_bbox_to_page(
                        crop_bbox=line_bbox,
                        region_bbox=bbox,
                        page_width=page_w,
                        page_height=page_h,
                    )
                    if page_bbox is None:
                        continue
                    page_ocr_records.append(
                        _build_record(
                            page=page_num,
                            record_type="ocr_text",
                            text=text,
                            bbox=page_bbox,
                            timing_ms=0,
                            block_type=region["block_type"],
                            confidence=confidence,
                            trace_region_id=region_id,
                        )
                    )

        debug_path = debug_dir / f"page_{page_num:04d}.json"
        debug_path.write_text(json.dumps(debug_payload, ensure_ascii=True, indent=2), encoding="utf-8")

        final_timing_ms = int(round((time.perf_counter() - page_start) * 1000))
        for record in page_layout_records:
            record["timing_ms"] = final_timing_ms
        for record in page_ocr_records:
            record["timing_ms"] = final_timing_ms

        records.extend(page_layout_records)
        records.extend(page_ocr_records)

    return records
