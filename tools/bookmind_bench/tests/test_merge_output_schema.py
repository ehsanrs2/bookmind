import json
import hashlib
import re
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
                "meta": {
                    "confidence": 0.77,
                    "trace": {
                        "layout_region_id": "page_0001_region_007",
                        "layout_line_id": 123,
                    },
                },
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
    document_ids_first = [row["document_id"] for row in rows]
    chunk_ids_first = [row["chunk_id"] for row in rows]
    region_indexes_first = [row["region_index"] for row in rows]

    stats_second = merge_outputs(
        paddle_jsonl=paddle_path,
        vlm_jsonl=vlm_path,
        out_jsonl=out_path,
        min_text_chars=20,
    )
    rows_second = _read_jsonl(out_path)
    stable_ids_second = [row["stable_id"] for row in rows_second]
    document_ids_second = [row["document_id"] for row in rows_second]
    chunk_ids_second = [row["chunk_id"] for row in rows_second]
    region_indexes_second = [row["region_index"] for row in rows_second]

    assert stats.total_emitted == 3
    assert stats.figure_captions_matched == 1
    assert stats_second.total_emitted == 3
    assert stable_ids_first == stable_ids_second
    assert document_ids_first == document_ids_second
    assert chunk_ids_first == chunk_ids_second
    assert region_indexes_first == region_indexes_second

    required_keys = {
        "document_id",
        "source_id",
        "chunk_id",
        "stable_id",
        "page",
        "page_index",
        "region_index",
        "reading_order",
        "content_type",
        "text",
        "bbox",
        "meta",
    }
    for row in rows:
        assert required_keys.issubset(row.keys())
        assert {
            "pdf_page_start",
            "pdf_page_end",
            "source_engines",
            "block_type",
            "confidence",
            "trace",
        }.issubset(set(row["meta"].keys()))
        assert {"paddleocr_line_ids", "vlm_line_id"}.issubset(set(row["meta"]["trace"].keys()))
        assert isinstance(row["stable_id"], str)
        assert len(row["stable_id"]) == 16
        assert row["page_index"] == row["page"]
        assert isinstance(row["region_index"], int)
        assert row["region_index"] >= 1
        assert row["reading_order"] == row["region_index"]
        assert row["source_id"] == row["document_id"]
        assert isinstance(row["document_id"], str)
        assert len(row["document_id"]) == 40
        assert isinstance(row["chunk_id"], str)
        assert row["chunk_id"].startswith("v1_")
        assert len(row["chunk_id"]) == 43

    figure_rows = [row for row in rows if row["content_type"] == "figure_caption"]
    assert len(figure_rows) == 1
    figure_meta = figure_rows[0]["meta"]
    assert figure_meta["figure_ref"]["layout_region_id"] == "page_0001_region_007"
    assert figure_meta["figure_ref"]["layout_line_id"] == 123
    assert "figure_context" in figure_meta
    assert len(figure_meta["figure_context"]) <= 400

    # Stable ID must remain based only on page/type/bbox/text (not new meta fields).
    text_norm = re.sub(r"\s+", " ", figure_rows[0]["text"].strip())
    token = f"1|figure_caption|100,100,200,200|{text_norm}"
    expected_stable_id = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    assert figure_rows[0]["stable_id"] == expected_stable_id

    expected_document_id = hashlib.sha1(
        str(paddle_path.resolve()).replace("\\", "/").encode("utf-8")
    ).hexdigest()
    assert all(row["document_id"] == expected_document_id for row in rows)

    chunk_row = figure_rows[0]
    chunk_token = (
        f"{expected_document_id}|1|figure_caption|100,100,200,200|"
        f"{re.sub(r'\\s+', ' ', chunk_row['text'].strip())}"
    )
    expected_chunk_id = "v1_" + hashlib.sha1(chunk_token.encode("utf-8")).hexdigest()
    assert chunk_row["chunk_id"] == expected_chunk_id
