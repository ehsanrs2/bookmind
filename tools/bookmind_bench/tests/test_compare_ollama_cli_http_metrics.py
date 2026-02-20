from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_ollama_cli_http.py"
    spec = importlib.util.spec_from_file_location("compare_ollama_cli_http_test", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_chat_metrics_empty_content_with_thinking() -> None:
    module = _load_module()
    payload = {
        "message": {"content": "", "thinking": "hidden reasoning"},
        "done_reason": "length",
        "eval_count": 256,
        "prompt_eval_count": 33,
    }
    out = module.extract_chat_metrics(payload)
    assert out["non_empty_final"] is False
    assert out["thinking_present"] is True
    assert out["done_reason"] == "length"
    assert out["eval_count"] == 256
    assert out["prompt_eval_count"] == 33


def test_extract_generate_metrics_empty_response_with_thinking() -> None:
    module = _load_module()
    payload = {
        "response": "  ",
        "thinking": "hidden reasoning",
        "done_reason": "length",
        "eval_count": 512,
        "prompt_eval_count": 21,
    }
    out = module.extract_generate_metrics(payload)
    assert out["non_empty_final"] is False
    assert out["thinking_present"] is True
    assert out["done_reason"] == "length"
    assert out["eval_count"] == 512
    assert out["prompt_eval_count"] == 21
