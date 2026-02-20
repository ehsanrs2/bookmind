from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import requests

from engines.vlm_providers import OllamaProvider
from rag_preview import generate_answer


def _load_debug_runner():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "debug_ollama_thinking.py"
    )
    spec = importlib.util.spec_from_file_location("bookmind_ollama_debug_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load debug script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_ollama_debug


def test_generate_answer_does_not_accept_empty_content_with_thinking(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self):
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "message": {"content": "", "thinking": "detailed reasoning " * 40},
                "done_reason": "length",
                "eval_count": 512,
                "prompt_eval_count": 32,
            }

    calls = []

    def _fake_post(url, json=None, timeout=60):
        assert url == "http://127.0.0.1:11434/api/chat"
        assert json is not None
        calls.append(json)
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)
    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="text",
    )
    with pytest.raises(RuntimeError) as exc_info:
        generate_answer(
            provider=provider,
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "Q"}],
            ollama_num_predict=512,
            ollama_retry_num_predict=1024,
            ollama_think=False,
        )
    message = str(exc_info.value)
    assert "empty assistant content with think=false" in message
    assert "done_reason=length" in message
    assert "eval_count=512" in message
    assert "prompt_eval_count=32" in message
    assert len(calls) == 3
    assert calls[0]["think"] is False
    assert calls[1]["options"]["num_predict"] >= 1024


def test_generate_answer_length_retry_recovers_with_ultra_short_fallback(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, payload):
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    payloads = [
        {
            "message": {"content": "", "thinking": "reasoning 1"},
            "done_reason": "length",
            "eval_count": 512,
            "prompt_eval_count": 21,
        },
        {
            "message": {"content": "", "thinking": "reasoning 2"},
            "done_reason": "length",
            "eval_count": 1024,
            "prompt_eval_count": 22,
        },
        {
            "message": {"content": '{"answer":"fallback succeeded"}'},
            "done_reason": "stop",
            "eval_count": 200,
            "prompt_eval_count": 15,
        },
    ]
    call_idx = {"value": 0}
    requests_seen = []

    def _fake_post(url, json=None, timeout=60):
        assert json is not None
        idx = call_idx["value"]
        call_idx["value"] += 1
        requests_seen.append(json)
        return _FakeResponse(payloads[idx])

    monkeypatch.setattr(requests, "post", _fake_post)
    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="text",
    )
    answer, usage = generate_answer(
        provider=provider,
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "Q"}],
        ollama_num_predict=512,
        ollama_retry_num_predict=1024,
        ollama_think=False,
    )
    assert answer == "fallback succeeded"
    assert usage is not None
    assert usage["attempt_name"] == "ultra_short_json"
    assert requests_seen[0]["options"]["num_predict"] == 512
    assert requests_seen[1]["options"]["num_predict"] >= 1024
    assert requests_seen[2]["options"]["num_predict"] >= 1024


def test_ollama_integration_repro_modes(tmp_path: Path) -> None:
    if os.getenv("BOOKMIND_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set BOOKMIND_OLLAMA_INTEGRATION=1 to run local Ollama integration checks.")

    fixture = Path(__file__).resolve().parent / "fixtures" / "figure3_tiny.png"
    if not fixture.exists():
        pytest.skip(f"Fixture missing: {fixture}")

    run_ollama_debug = _load_debug_runner()

    text_summary = run_ollama_debug(
        image=str(fixture),
        prompt="What does this figure show?",
        format_mode="text",
        think=False,
        num_predict=256,
        trials=3,
        out_dir=str(tmp_path / "text_mode"),
        temperature=0.0,
    )
    text_counts = text_summary["counts"]
    assert text_counts["http_error"] == 0
    assert text_counts["non_empty_content"] >= 1

    strict_summary = run_ollama_debug(
        image=str(fixture),
        prompt="What does this figure show?",
        format_mode="json-in-text",
        think=False,
        num_predict=256,
        trials=5,
        out_dir=str(tmp_path / "strict_mode"),
        temperature=0.0,
    )
    strict_counts = strict_summary["counts"]
    assert strict_counts["http_error"] == 0
    if strict_counts["empty_content_with_thinking"] == 0:
        pytest.skip(
            "Strict json-in-text did not reproduce empty-content-with-thinking in this run; "
            "re-run integration for reproduction stats."
        )
