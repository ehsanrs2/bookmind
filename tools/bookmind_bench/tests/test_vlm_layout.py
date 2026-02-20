import json
from pathlib import Path

from PIL import Image

from engines.vlm_providers import EmptyCaptionOutputError
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

    class _FakeProvider:
        def caption(self, image_path: str, prompt: str, max_tokens: int, temperature: float):
            return "caption one", None

    monkeypatch.setattr(
        "engines.vlm_caption_engine.build_vlm_provider",
        lambda **kwargs: _FakeProvider(),
    )

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

    class _FakeProvider:
        def caption(self, image_path: str, prompt: str, max_tokens: int, temperature: float):
            seen_prompt["value"] = prompt
            return "caption two", None

    monkeypatch.setattr(
        "engines.vlm_caption_engine.build_vlm_provider",
        lambda **kwargs: _FakeProvider(),
    )
    run_vlm_layout(layout_jsonl=str(layout_path), imgdir=str(imgdir), out_dir=str(tmp_path))

    prompt = seen_prompt["value"]
    assert "REGION SPECIFIC" in prompt
    assert "PAGE WIDE NOISE" not in prompt


def test_vlm_layout_output_schema_unchanged(tmp_path: Path, monkeypatch) -> None:
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
            }
        ],
    )

    class _FakeProvider:
        def caption(self, image_path: str, prompt: str, max_tokens: int, temperature: float):
            return "caption", None

    monkeypatch.setattr(
        "engines.vlm_caption_engine.build_vlm_provider",
        lambda **kwargs: _FakeProvider(),
    )
    run_vlm_layout(layout_jsonl=str(layout_path), imgdir=str(imgdir), out_dir=str(tmp_path))

    row = _read_jsonl(tmp_path / "vlm" / "output.jsonl")[0]
    assert set(row.keys()) == {"engine", "page", "type", "text", "bbox", "timing_ms", "meta"}
    assert row["engine"] == "vlm"
    assert row["type"] == "figure_caption"


def test_vlm_layout_empty_first_attempt_retries_and_writes_debug(tmp_path: Path, monkeypatch) -> None:
    imgdir = tmp_path / "images"
    imgdir.mkdir(parents=True)
    Image.new("RGB", (120, 120), "white").save(imgdir / "page_0001.png")

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
            }
        ],
    )

    class _RetryProvider:
        def __init__(self):
            self.calls = 0

        def caption(self, image_path: str, prompt: str, max_tokens: int, temperature: float):
            self.calls += 1
            if self.calls == 1:
                raise EmptyCaptionOutputError(
                    "empty",
                    {
                        "endpoint": "/api/chat",
                        "status_code": 200,
                        "timing_ms": 123,
                        "done_reason": "length",
                        "eval_count": 256,
                        "prompt_eval_count": 100,
                        "raw_head_500": "{\"message\":{\"content\":\"\"}}",
                        "parsed_keys": ["message", "done_reason"],
                        "response_keys": ["message", "done_reason"],
                    },
                )
            return "Recovered caption", {
                "eval_count": 4,
                "debug_info": {
                    "endpoint": "/api/generate",
                    "status_code": 200,
                    "timing_ms": 44,
                    "done_reason": "stop",
                    "eval_count": 600,
                    "prompt_eval_count": 22,
                    "raw_head_500": "{\"response\":\"Recovered caption\"}",
                    "parsed_keys": ["response"],
                    "response_keys": ["response"],
                },
            }

    provider = _RetryProvider()
    monkeypatch.setattr("engines.vlm_caption_engine.build_vlm_provider", lambda **kwargs: provider)

    written = run_vlm_layout(layout_jsonl=str(layout_path), imgdir=str(imgdir), out_dir=str(tmp_path))
    assert written == 1

    row = _read_jsonl(tmp_path / "vlm" / "output.jsonl")[0]
    assert row["text"] == "Recovered caption"

    debug_dir = tmp_path / "vlm" / "debug"
    attempt_req = debug_dir / "caption_p1_idx1_attempt1_request.json"
    attempt_rsp = debug_dir / "caption_p1_idx1_attempt1_response.json"
    retry_req = debug_dir / "caption_p1_idx1_retry_request.json"
    retry_rsp = debug_dir / "caption_p1_idx1_retry_response.json"
    assert attempt_req.exists()
    assert attempt_rsp.exists()
    assert retry_req.exists()
    assert retry_rsp.exists()
    req = json.loads(attempt_req.read_text(encoding="utf-8"))
    rsp = json.loads(attempt_rsp.read_text(encoding="utf-8"))
    assert req["max_tokens"] == 256
    assert rsp["done_reason"] == "length"
