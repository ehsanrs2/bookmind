"""VLM planning hook for region-aware figure captioning."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Dict, List

from engines.paddleocr_engine import list_page_images


def _ensure_image_map(imgdir: str) -> Dict[int, Path]:
    images = list_page_images(imgdir)
    return {page: path for page, path in images}


def _suggest_crop_path(out_jobs_path: Path, page: int, index: int) -> Path:
    base_dir = out_jobs_path.parent / "vlm_crops"
    return base_dir / f"page_{page:04d}_block_{index:03d}.png"


def _is_figure_record(record: Dict[str, Any]) -> bool:
    block_type = ""
    meta = record.get("meta")
    if isinstance(meta, dict):
        block_type = str(meta.get("block_type", ""))
    if block_type.lower() == "figure":
        return True
    return record.get("type") == "layout_block" and block_type.lower() == "figure"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plan_vlm_jobs(records_jsonl_path: str, imgdir: str, out_jobs_path: str) -> int:
    records_path = Path(records_jsonl_path)
    out_path = Path(out_jobs_path)
    image_map = _ensure_image_map(imgdir)

    jobs: List[Dict[str, Any]] = []
    figure_index = 0
    for record in _read_jsonl(records_path):
        if not isinstance(record, dict):
            continue
        if not _is_figure_record(record):
            continue
        page = record.get("page")
        if not isinstance(page, int):
            continue
        bbox = record.get("bbox")
        image_path = image_map.get(page)
        if image_path is None:
            continue
        figure_index += 1
        jobs.append(
            {
                "page": page,
                "bbox": bbox,
                "image_path": image_path.as_posix(),
                "suggested_crop_path": _suggest_crop_path(out_path, page, figure_index).as_posix(),
                "prompt_template": "technical_diagram_v1",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=True))
            handle.write("\n")

    return len(jobs)
