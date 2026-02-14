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


def test_merge_layout_with_vlm(tmp_path: Path) -> None:
    layout_path = tmp_path / "layout" / "output.jsonl"
    vlm_path = tmp_path / "vlm" / "output.jsonl"
    out_path = tmp_path / "ingest" / "records.jsonl"

    _write_jsonl(
        layout_path,
        [
            {
                "engine": "layoutparser",
                "page": 1,
                "type": "layout_block",
                "text": "",
                "bbox": [100, 100, 200, 200],
                "meta": {"block_type": "figure", "region_id": "page_0001_region_001"},
            },
            {
                "engine": "layoutparser",
                "page": 1,
                "type": "ocr_text",
                "text": "A long enough OCR text line from layout parser.",
                "bbox": [10, 10, 80, 30],
                "meta": {"block_type": "text", "trace": {"region_id": "page_0001_region_009"}},
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
                "text": "Figure caption from vlm-layout",
                "bbox": [100, 100, 200, 200],
                "meta": {
                    "trace": {
                        "layout_region_id": "page_0001_region_001",
                        "layout_line_id": 1,
                    },
                    "page_image": "images/page_0001.png",
                    "crop_image": "vlm/crops/page_0001_line_00001.png",
                },
            }
        ],
    )

    stats = merge_outputs(
        paddle_jsonl=layout_path,
        vlm_jsonl=vlm_path,
        out_jsonl=out_path,
        min_text_chars=20,
    )
    rows = _read_jsonl(out_path)
    text_rows = [row for row in rows if row["content_type"] == "text"]
    figure_rows = [row for row in rows if row["content_type"] == "figure_caption"]

    assert stats.total_emitted == 2
    assert stats.figure_captions_matched == 1
    assert len(text_rows) == 1
    assert len(figure_rows) == 1
    assert text_rows[0]["meta"]["source_engines"] == ["layoutparser"]
    assert text_rows[0]["meta"]["trace"]["region_id"] == "page_0001_region_009"
    assert figure_rows[0]["meta"]["source_engines"] == ["layoutparser", "vlm"]
    assert figure_rows[0]["meta"]["trace"]["region_id"] == "page_0001_region_001"
    assert figure_rows[0]["meta"]["trace"]["layout_line_id"] == 1
    assert figure_rows[0]["meta"]["figure_ref"]["layout_region_id"] == "page_0001_region_001"
    assert figure_rows[0]["meta"]["figure_ref"]["layout_line_id"] == 1
    assert figure_rows[0]["meta"]["figure_ref"]["page_image"] == "images/page_0001.png"
    assert figure_rows[0]["meta"]["figure_ref"]["crop_image"] == "vlm/crops/page_0001_line_00001.png"
    assert len(figure_rows[0]["meta"]["figure_context"]) <= 400
    assert "A long enough OCR text line from layout parser." in figure_rows[0]["meta"]["figure_context"]
