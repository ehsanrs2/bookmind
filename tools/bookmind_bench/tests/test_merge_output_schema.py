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


def test_merge_output_schema_and_stable_ids(tmp_path: Path) -> None:
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
                "bbox": [100, 100, 200, 200],
                "meta": {"block_type": "figure", "confidence": 0.91},
            },
            {
                "engine": "paddleocr",
                "page": 2,
                "type": "ocr_text",
                "text": "A compact but valid OCR paragraph for page two.",
                "bbox": [10.2, 20.2, 220.8, 80.7],
                "meta": {"block_type": "text", "confidence": 0.88},
            },
            {
                "engine": "paddleocr",
                "page": 2,
                "type": "ocr_text",
                "text": "tiny",
                "bbox": [1, 1, 2, 2],
                "meta": {"block_type": "text"},
            },
            {
                "engine": "paddleocr",
                "page": 2,
                "type": "table_md",
                "text": "| Col |\n| --- |\n| v |",
                "bbox": [30, 90, 230, 180],
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
                "text": "Figure caption text",
                "bbox": [100, 100, 200, 200],
                "meta": {"confidence": 0.77},
            }
        ],
    )

    stats = merge_outputs(
        paddle_jsonl=paddle_path,
        vlm_jsonl=vlm_path,
        out_jsonl=out_path,
        min_text_chars=20,
    )
    rows = _read_jsonl(out_path)
    stable_ids_first = [row["stable_id"] for row in rows]

    stats_second = merge_outputs(
        paddle_jsonl=paddle_path,
        vlm_jsonl=vlm_path,
        out_jsonl=out_path,
        min_text_chars=20,
    )
    rows_second = _read_jsonl(out_path)
    stable_ids_second = [row["stable_id"] for row in rows_second]

    assert stats.total_emitted == 3
    assert stats.figure_captions_matched == 1
    assert stats_second.total_emitted == 3
    assert stable_ids_first == stable_ids_second

    required_keys = {"stable_id", "page", "content_type", "text", "bbox", "meta"}
    for row in rows:
        assert required_keys.issubset(row.keys())
        assert set(row["meta"].keys()) == {
            "pdf_page_start",
            "pdf_page_end",
            "source_engines",
            "block_type",
            "confidence",
            "trace",
        }
        assert set(row["meta"]["trace"].keys()) == {"paddleocr_line_ids", "vlm_line_id"}
        assert isinstance(row["stable_id"], str)
        assert len(row["stable_id"]) == 16
