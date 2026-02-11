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
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def test_merge_prefers_exact_bbox_match(tmp_path: Path) -> None:
    paddle_path = tmp_path / "paddleocr" / "output.jsonl"
    vlm_path = tmp_path / "vlm" / "output.jsonl"
    out_path = tmp_path / "ingest" / "records.jsonl"

    _write_jsonl(
        paddle_path,
        [
            {
                "engine": "paddleocr",
                "page": 1,
                "type": "layout_block",
                "text": "",
                "bbox": [10, 10, 50, 50],
                "meta": {"block_type": "figure"},
            },
            {
                "engine": "paddleocr",
                "page": 2,
                "type": "ocr_text",
                "text": "This is enough text to pass min chars.",
                "bbox": [5, 5, 30, 18],
                "meta": {"block_type": "text"},
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
                "text": "Exact caption",
                "bbox": [10, 10, 50, 50],
            },
            {
                "engine": "vlm",
                "page": 1,
                "type": "figure_caption",
                "text": "Tolerance caption",
                "bbox": [11, 11, 51, 51],
            },
        ],
    )

    stats = merge_outputs(paddle_jsonl=paddle_path, vlm_jsonl=vlm_path, out_jsonl=out_path)
    rows = _read_jsonl(out_path)
    figure_rows = [row for row in rows if row["content_type"] == "figure_caption"]

    assert stats.figure_captions_matched == 1
    assert len(figure_rows) == 1
    assert figure_rows[0]["text"] == "Exact caption"
    assert figure_rows[0]["meta"]["trace"]["vlm_line_id"] == 1


def test_merge_matches_with_bbox_tolerance_when_iou_below_threshold(tmp_path: Path) -> None:
    paddle_path = tmp_path / "paddleocr" / "output.jsonl"
    vlm_path = tmp_path / "vlm" / "output.jsonl"
    out_path = tmp_path / "ingest" / "records.jsonl"

    _write_jsonl(
        paddle_path,
        [
            {
                "engine": "paddleocr",
                "page": 1,
                "type": "layout_block",
                "text": "",
                "bbox": [10, 10, 50, 50],
                "meta": {"block_type": "figure"},
            },
            {
                "engine": "paddleocr",
                "page": 2,
                "type": "table_md",
                "text": "|A|B|",
                "bbox": [0, 0, 20, 20],
                "meta": {"block_type": "table"},
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
                "text": "Tolerance-only caption",
                "bbox": [11, 11, 51, 51],
            }
        ],
    )

    stats = merge_outputs(
        paddle_jsonl=paddle_path,
        vlm_jsonl=vlm_path,
        out_jsonl=out_path,
        bbox_tol=2,
        iou_threshold=0.95,
    )
    rows = _read_jsonl(out_path)
    figure_rows = [row for row in rows if row["content_type"] == "figure_caption"]

    assert stats.figure_captions_matched == 1
    assert len(figure_rows) == 1
    assert figure_rows[0]["text"] == "Tolerance-only caption"
