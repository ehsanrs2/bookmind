"""Create portable ingest bundles for bench run directories."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class BundleResult:
    bundle_dir: Path
    records_count: int
    copied_images: int
    missing_images: int


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def _normalize_bbox(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out: List[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        out.append(float(item))
    return [min(out[0], out[2]), min(out[1], out[3]), max(out[0], out[2]), max(out[1], out[3])]


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


def _resolve_existing_path(
    raw_path: str,
    *,
    run_dir: Path,
    imgdir: Optional[Path],
    extra_roots: Optional[List[Path]] = None,
) -> Optional[Path]:
    path_value = str(raw_path).strip()
    if not path_value:
        return None

    candidates: List[Path] = []
    raw = Path(path_value)
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(run_dir / raw)
        if imgdir is not None:
            candidates.append(imgdir / raw)
            candidates.append(imgdir / raw.name)
    if extra_roots:
        for root in extra_roots:
            candidates.append(root / raw)
            candidates.append(root / raw.name)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _build_vlm_crop_candidates(run_dir: Path, imgdir: Optional[Path]) -> List[Dict[str, Any]]:
    vlm_jsonl = run_dir / "vlm" / "output.jsonl"
    if not vlm_jsonl.exists():
        return []

    png_paths = [p for p in (run_dir / "vlm").rglob("*.png") if p.is_file()]
    by_basename: Dict[str, List[Path]] = defaultdict(list)
    for png in png_paths:
        by_basename[png.name].append(png.resolve())

    rows = _read_jsonl(vlm_jsonl)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("type") != "figure_caption":
            continue
        page = row.get("page")
        bbox = _normalize_bbox(row.get("bbox"))
        if not isinstance(page, int) or bbox is None:
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        trace = meta.get("trace") if isinstance(meta.get("trace"), dict) else {}

        crop_path: Optional[Path] = None
        for key in ("crop_image", "crop_image_path"):
            raw_crop = meta.get(key)
            if isinstance(raw_crop, str):
                crop_path = _resolve_existing_path(
                    raw_crop,
                    run_dir=run_dir,
                    imgdir=imgdir,
                    extra_roots=[run_dir / "vlm"],
                )
                if crop_path is not None:
                    break

        if crop_path is None:
            layout_line_id = trace.get("layout_line_id")
            if isinstance(layout_line_id, int):
                expected = f"page_{page:04d}_line_{layout_line_id:05d}.png"
                matches = by_basename.get(expected, [])
                if len(matches) == 1:
                    crop_path = matches[0]

        if crop_path is None:
            continue
        out.append({"page": page, "bbox": bbox, "path": crop_path})
    return out


def _derive_crop_path(
    record: Dict[str, Any],
    vlm_candidates: Sequence[Dict[str, Any]],
) -> Optional[Path]:
    page = record.get("page")
    bbox = _normalize_bbox(record.get("bbox"))
    if not isinstance(page, int) or bbox is None:
        return None

    best_path: Optional[Path] = None
    best_iou = 0.0
    for item in vlm_candidates:
        if item.get("page") != page:
            continue
        candidate_bbox = item.get("bbox")
        candidate_path = item.get("path")
        if not isinstance(candidate_bbox, list) or not isinstance(candidate_path, Path):
            continue
        iou = _bbox_iou(bbox, candidate_bbox)
        if iou > best_iou:
            best_iou = iou
            best_path = candidate_path
    if best_iou <= 0.0:
        return None
    return best_path


def _build_page_image_from_imgdir(page: Any, imgdir: Optional[Path]) -> Optional[Path]:
    if imgdir is None or not isinstance(page, int):
        return None
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = imgdir / f"page_{page:04d}{suffix}"
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _copy_image_into_bundle(
    source: Path,
    *,
    kind: str,
    bundle_images_dir: Path,
    copied_cache: Dict[str, str],
) -> str:
    source_key = str(source.resolve())
    cached = copied_cache.get(source_key)
    if cached is not None:
        return cached

    digest = sha1(source_key.encode("utf-8")).hexdigest()[:10]
    destination_name = f"{source.stem}_{digest}{source.suffix.lower() or '.bin'}"
    relative = Path("images") / kind / destination_name
    destination = bundle_images_dir.parent / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    rel_text = relative.as_posix()
    copied_cache[source_key] = rel_text
    return rel_text


def _build_report(records: Sequence[Dict[str, Any]], missing_images: int) -> Dict[str, Any]:
    total_records = len(records)
    counts_by_content_type = Counter(
        str(row.get("content_type")) for row in records if isinstance(row.get("content_type"), str)
    )

    figure_rows = [row for row in records if row.get("content_type") == "figure_caption"]
    text_rows = [row for row in records if row.get("content_type") == "text"]

    caption_lengths = [len(str(row.get("text") or "")) for row in figure_rows]
    context_lengths: List[int] = []
    with_crop_image = 0
    with_page_image = 0
    for row in figure_rows:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        figure_ref = meta.get("figure_ref") if isinstance(meta.get("figure_ref"), dict) else {}
        figure_context = str(meta.get("figure_context") or "")
        context_lengths.append(len(figure_context))
        if isinstance(figure_ref.get("crop_image"), str) and figure_ref.get("crop_image").strip():
            with_crop_image += 1
        if isinstance(figure_ref.get("page_image"), str) and figure_ref.get("page_image").strip():
            with_page_image += 1

    text_lengths = [len(str(row.get("text") or "")) for row in text_rows]
    per_page: Dict[int, Dict[str, int]] = {}
    for row in records:
        page = row.get("page")
        content_type = row.get("content_type")
        if not isinstance(page, int) or not isinstance(content_type, str):
            continue
        page_counts = per_page.setdefault(page, {})
        page_counts[content_type] = int(page_counts.get(content_type, 0)) + 1

    return {
        "total_records": total_records,
        "counts_by_content_type": dict(counts_by_content_type),
        "figure_caption": {
            "captions_total": len(figure_rows),
            "avg_caption_chars": (sum(caption_lengths) / len(caption_lengths)) if caption_lengths else 0.0,
            "avg_context_chars": (sum(context_lengths) / len(context_lengths)) if context_lengths else 0.0,
            "with_crop_image": with_crop_image,
            "with_page_image": with_page_image,
            "missing_images": missing_images,
        },
        "text": {
            "avg_text_chars": (sum(text_lengths) / len(text_lengths)) if text_lengths else 0.0,
            "p50_text_chars": median(text_lengths) if text_lengths else 0,
        },
        "per_page": per_page,
    }


def _build_report_md(manifest: Dict[str, Any], report: Dict[str, Any]) -> str:
    counts = report.get("counts_by_content_type", {})
    fig = report.get("figure_caption", {})
    txt = report.get("text", {})
    images = manifest.get("images", {})
    lines = [
        "# Bundle Quality Report",
        "",
        f"- Total records: {report.get('total_records', 0)}",
        (
            f"- Content types: text={counts.get('text', 0)}, "
            f"table={counts.get('table', 0)}, "
            f"figure_caption={counts.get('figure_caption', 0)}"
        ),
        (
            f"- Figure captions: total={fig.get('captions_total', 0)}, "
            f"avg_caption_chars={fig.get('avg_caption_chars', 0):.2f}, "
            f"avg_context_chars={fig.get('avg_context_chars', 0):.2f}"
        ),
        (
            f"- Text quality: avg_text_chars={txt.get('avg_text_chars', 0):.2f}, "
            f"p50_text_chars={txt.get('p50_text_chars', 0)}"
        ),
        (
            f"- Images: included={images.get('included', False)}, "
            f"copied={images.get('count', 0)}, missing={images.get('missing', 0)}"
        ),
    ]
    return "\n".join(lines) + "\n"


def bundle_run(
    *,
    run_dir: str | Path,
    imgdir: str | Path | None = None,
    include_images: bool = False,
) -> BundleResult:
    run_dir_path = Path(run_dir)
    ingest_path = run_dir_path / "ingest" / "records.jsonl"
    if not ingest_path.exists():
        raise SystemExit(f"Ingest records JSONL not found: {ingest_path}")

    imgdir_path = Path(imgdir) if imgdir is not None else None
    if imgdir_path is not None and not imgdir_path.exists():
        raise SystemExit(f"Image directory not found: {imgdir_path}")

    bundle_dir = run_dir_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_records_path = bundle_dir / "records.jsonl"
    bundle_manifest_path = bundle_dir / "manifest.json"
    bundle_report_json_path = bundle_dir / "report.json"
    bundle_report_md_path = bundle_dir / "report.md"
    bundle_images_dir = bundle_dir / "images"

    records = _read_jsonl(ingest_path)
    copied_cache: Dict[str, str] = {}
    copied_images = 0
    missing_images = 0
    vlm_candidates = _build_vlm_crop_candidates(run_dir_path, imgdir_path) if include_images else []

    if include_images:
        bundle_images_dir.mkdir(parents=True, exist_ok=True)

    for row in records:
        if row.get("content_type") != "figure_caption":
            continue
        meta = row.get("meta")
        if not isinstance(meta, dict):
            continue
        figure_ref = meta.get("figure_ref")
        if not isinstance(figure_ref, dict):
            continue

        page = row.get("page")
        page_image = figure_ref.get("page_image")
        crop_image = figure_ref.get("crop_image")

        if include_images:
            page_source: Optional[Path] = None
            if isinstance(page_image, str):
                page_source = _resolve_existing_path(
                    page_image,
                    run_dir=run_dir_path,
                    imgdir=imgdir_path,
                    extra_roots=[run_dir_path / "images"],
                )
                if page_source is None:
                    missing_images += 1
            else:
                page_source = _build_page_image_from_imgdir(page, imgdir_path)

            if page_source is not None:
                before = len(copied_cache)
                figure_ref["page_image"] = _copy_image_into_bundle(
                    page_source,
                    kind="pages",
                    bundle_images_dir=bundle_images_dir,
                    copied_cache=copied_cache,
                )
                copied_images += max(0, len(copied_cache) - before)

            crop_source: Optional[Path] = None
            if isinstance(crop_image, str):
                crop_source = _resolve_existing_path(
                    crop_image,
                    run_dir=run_dir_path,
                    imgdir=imgdir_path,
                    extra_roots=[run_dir_path / "vlm"],
                )
                if crop_source is None:
                    missing_images += 1
            else:
                crop_source = _derive_crop_path(row, vlm_candidates)

            if crop_source is not None:
                before = len(copied_cache)
                figure_ref["crop_image"] = _copy_image_into_bundle(
                    crop_source,
                    kind="crops",
                    bundle_images_dir=bundle_images_dir,
                    copied_cache=copied_cache,
                )
                copied_images += max(0, len(copied_cache) - before)

    _write_jsonl(bundle_records_path, records)

    content_types = Counter(
        str(row.get("content_type")) for row in records if isinstance(row.get("content_type"), str)
    )
    manifest = {
        "run_dir": str(run_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": {
            "path": "records.jsonl",
            "count": len(records),
            "content_types": {
                "text": int(content_types.get("text", 0)),
                "table": int(content_types.get("table", 0)),
                "figure_caption": int(content_types.get("figure_caption", 0)),
            },
        },
        "sources": {
            "layout_jsonl": "../layout/output.jsonl" if (run_dir_path / "layout" / "output.jsonl").exists() else None,
            "paddleocr_jsonl": "../paddleocr/output.jsonl"
            if (run_dir_path / "paddleocr" / "output.jsonl").exists()
            else None,
            "vlm_jsonl": "../vlm/output.jsonl" if (run_dir_path / "vlm" / "output.jsonl").exists() else None,
        },
        "images": {
            "included": bool(include_images),
            "dir": "images" if include_images else None,
            "count": copied_images,
            "missing": missing_images,
        },
    }

    report = _build_report(records, missing_images)
    report_md = _build_report_md(manifest, report)

    _write_json(bundle_manifest_path, manifest)
    _write_json(bundle_report_json_path, report)
    bundle_report_md_path.write_text(report_md, encoding="utf-8")

    return BundleResult(
        bundle_dir=bundle_dir,
        records_count=len(records),
        copied_images=copied_images,
        missing_images=missing_images,
    )
