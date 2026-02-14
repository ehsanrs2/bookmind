from __future__ import annotations

import json
from pathlib import Path

import requests

from engines.vlm_providers import (
    OllamaProvider,
    OpenAICompatProvider,
    build_vlm_provider,
)


def test_openai_compat_provider_caption(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "crop.png"
    image.write_bytes(b"png-bytes")

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            payload = {
                "choices": [{"message": {"content": "openai caption"}}],
                "usage": {"total_tokens": 10},
            }
            return json.dumps(payload).encode("utf-8")

    def _fake_urlopen(request_obj, timeout=60):
        assert request_obj.full_url == "http://127.0.0.1:8000/v1/chat/completions"
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    provider = OpenAICompatProvider(endpoint="http://127.0.0.1:8000/v1", model="qwen3-vl")
    text, usage = provider.caption(
        image_path=str(image),
        prompt="Describe this figure",
        max_tokens=64,
        temperature=0.1,
    )
    assert text == "openai caption"
    assert usage == {"total_tokens": 10}


def test_ollama_provider_caption(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "crop.png"
    image.write_bytes(b"png-bytes")

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "message": {"content": "ollama caption"},
                "prompt_eval_count": 1,
                "eval_count": 2,
                "prompt_eval_duration": 3,
                "eval_duration": 4,
                "total_duration": 5,
            }

    def _fake_post(url, json=None, timeout=60):
        assert url == "http://127.0.0.1:11434/api/chat"
        assert json is not None
        assert json["model"] == "qwen3-vl:latest"
        assert json["messages"][0]["role"] == "user"
        assert json["messages"][0]["images"]
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

    provider = OllamaProvider(ollama_url="http://127.0.0.1:11434", model="qwen3-vl:latest")
    text, usage = provider.caption(
        image_path=str(image),
        prompt="Describe this figure",
        max_tokens=64,
        temperature=0.1,
    )
    assert text == "ollama caption"
    assert usage is not None
    assert usage["eval_count"] == 2
    assert "timing_ms" in usage


def test_provider_factory_backend_switch() -> None:
    vllm_provider = build_vlm_provider(
        backend="vllm",
        endpoint="http://127.0.0.1:8000/v1",
        model="qwen3-vl",
        ollama_url="http://127.0.0.1:11434",
        ollama_model="qwen3-vl:latest",
    )
    ollama_provider = build_vlm_provider(
        backend="ollama",
        endpoint="http://127.0.0.1:8000/v1",
        model="qwen3-vl",
        ollama_url="http://127.0.0.1:11434",
        ollama_model="qwen3-vl:latest",
    )
    assert isinstance(vllm_provider, OpenAICompatProvider)
    assert isinstance(ollama_provider, OllamaProvider)
