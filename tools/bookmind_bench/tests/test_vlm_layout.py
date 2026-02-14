import json
from pathlib import Path

from PIL import Image

from engines.vlm_caption_engine import run_vlm_layout


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_vlm_layout_selection(tmp_path: Path, monkeypatch) -> None:
    imgdir = tmp_path / "images"
    imgdir.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(imgdir / "page_0001.png")

    layout_path = tmp_path / "layout" / "output.jsonl"
    _write_jsonl(
        layout_path,
        [
            {
                "engine": "layoutparser",
                "page": 1,
                "type": "layout_block",
                "text": "",
                "bbox": [10, 10, 80, 80],
                "meta": {"block_type": "figure", "region_id": "page_0001_region_001"},
            },
            {
                "engine": "layoutparser",
                "page": 1,
                "type": "layout_block",
                "text": "",
                "bbox": [1, 1, 20, 20],
                "meta": {"block_type": "text", "region_id": "page_0001_region_002"},
            },
        ],
    )

    def _fake_post(_endpoint: str, _payload: dict, timeout_s: int = 60) -> dict:
        return {"choices": [{"message": {"content": "caption one"}}]}

    monkeypatch.setattr("engines.vlm_caption_engine._post_chat_completion", _fake_post)

    written = run_vlm_layout(
        layout_jsonl=str(layout_path),
        imgdir=str(imgdir),
        out_dir=str(tmp_path),
    )

    out_rows = _read_jsonl(tmp_path / "vlm" / "output.jsonl")
    assert written == 1
    assert len(out_rows) == 1
    assert out_rows[0]["bbox"] == [10, 10, 80, 80]
    assert out_rows[0]["meta"]["trace"]["layout_region_id"] == "page_0001_region_001"
    assert out_rows[0]["meta"]["trace"]["layout_line_id"] == 1


def test_vlm_layout_hinting(tmp_path: Path, monkeypatch) -> None:
    imgdir = tmp_path / "images"
    imgdir.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(imgdir / "page_0001.png")

    layout_path = tmp_path / "layout" / "output.jsonl"
    _write_jsonl(
        layout_path,
        [
            {
                "engine": "layoutparser",
                "page": 1,
                "type": "layout_block",
                "text": "",
                "bbox": [10, 10, 90, 90],
                "meta": {"block_type": "figure", "region_id": "page_0001_region_001"},
            },
            {
                "engine": "layoutparser",
                "page": 1,
                "type": "ocr_text",
                "text": "REGION SPECIFIC",
                "bbox": [20, 20, 50, 40],
                "meta": {"trace": {"region_id": "page_0001_region_001"}},
            },
            {
                "engine": "layoutparser",
                "page": 1,
                "type": "ocr_text",
                "text": "PAGE WIDE NOISE",
                "bbox": [1, 1, 5, 5],
                "meta": {"trace": {"region_id": "page_0001_region_999"}},
            },
        ],
    )

    seen_prompt: dict[str, str] = {}

    def _fake_post(_endpoint: str, payload: dict, timeout_s: int = 60) -> dict:
        seen_prompt["value"] = payload["messages"][0]["content"][0]["text"]
        return {"choices": [{"message": {"content": "caption two"}}]}

    monkeypatch.setattr("engines.vlm_caption_engine._post_chat_completion", _fake_post)
    run_vlm_layout(layout_jsonl=str(layout_path), imgdir=str(imgdir), out_dir=str(tmp_path))

    prompt = seen_prompt["value"]
    assert "REGION SPECIFIC" in prompt
    assert "PAGE WIDE NOISE" not in prompt

