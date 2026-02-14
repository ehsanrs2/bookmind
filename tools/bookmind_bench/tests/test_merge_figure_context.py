import json
from pathlib import Path

from merge import merge_outputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_merge_figure_context_uses_nearby_text_and_is_bounded(tmp_path: Path) -> None:
    paddle_path = tmp_path / "layout" / "output.jsonl"
    vlm_path = tmp_path / "vlm" / "output.jsonl"
    out_path = tmp_path / "ingest" / "records.jsonl"

    _write_jsonl(
        paddle_path,
        [
            {
                "engine": "layoutparser",
                "page": 1,
                "type": "ocr_text",
                "text": "Near figure label A and arrow explanation.",
                "bbox": [95, 90, 180, 120],
                "meta": {"block_type": "text"},
            },
            {
                "engine": "layoutparser",
                "page": 1,
                "type": "ocr_text",
                "text": "Near figure label B and output details.",
                "bbox": [120, 180, 210, 220],
                "meta": {"block_type": "text"},
            },
            {
                "engine": "layoutparser",
                "page": 1,
                "type": "layout_block",
                "text": "",
                "bbox": [100, 100, 200, 200],
                "meta": {"block_type": "figure", "region_id": "page_0001_region_002"},
            },
        ],
    )
    _write_jsonl(
        vlm_path,
        [
            {
                "engine": "vlm",
                "page": 1,
                "type": "figure_caption",
                "text": (
                    "This figure shows a controller path and sensor feedback loop. "
                    "The output is regulated through two processing blocks and then returned."
                ),
                "bbox": [100, 100, 200, 200],
                "meta": {
                    "trace": {
                        "layout_region_id": "page_0001_region_002",
                        "layout_line_id": 7,
                    }
                },
            }
        ],
    )

    merge_outputs(
        paddle_jsonl=paddle_path,
        vlm_jsonl=vlm_path,
        out_jsonl=out_path,
        min_text_chars=20,
    )
    rows = _read_jsonl(out_path)
    figure_row = next(row for row in rows if row["content_type"] == "figure_caption")

    assert figure_row["meta"]["figure_ref"]["layout_region_id"] == "page_0001_region_002"
    assert figure_row["meta"]["figure_ref"]["layout_line_id"] == 7
    context = figure_row["meta"]["figure_context"]
    assert len(context) <= 400
    assert "This figure shows a controller path" in context
    assert "Near figure label A" in context
