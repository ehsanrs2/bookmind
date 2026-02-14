from __future__ import annotations

import json

import requests

from engines.vlm_providers import OllamaProvider, OpenAICompatProvider
from rag_preview import generate_answer


def test_generate_answer_uses_custom_chat_provider() -> None:
    class _FakeProvider:
        def chat(self, messages, max_tokens, temperature):
            assert messages[0]["role"] == "system"
            assert max_tokens == 55
            assert temperature == 0.3
            return "fake answer", {"total_tokens": 10}

    text, usage = generate_answer(
        provider=_FakeProvider(),
        messages=[{"role": "system", "content": "x"}],
        max_tokens=55,
        temperature=0.3,
    )

    assert text == "fake answer"
    assert usage == {"total_tokens": 10}


def test_generate_answer_openai_compat(monkeypatch) -> None:
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            payload = {
                "choices": [{"message": {"content": "openai rag answer"}}],
                "usage": {"total_tokens": 42},
            }
            return json.dumps(payload).encode("utf-8")

    def _fake_urlopen(request_obj, timeout=60):
        assert request_obj.full_url == "http://127.0.0.1:8000/v1/chat/completions"
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    provider = OpenAICompatProvider(endpoint="http://127.0.0.1:8000/v1", model="qwen3-vl")
    text, usage = generate_answer(
        provider=provider,
        messages=[{"role": "user", "content": "Q"}],
        max_tokens=77,
        temperature=0.1,
    )

    assert text == "openai rag answer"
    assert usage == {"total_tokens": 42}


def test_generate_answer_ollama(monkeypatch) -> None:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "message": {"content": "ollama rag answer"},
                "prompt_eval_count": 11,
                "eval_count": 12,
            }

    def _fake_post(url, json=None, timeout=60):
        assert url == "http://127.0.0.1:11434/api/chat"
        assert json is not None
        assert json["messages"][0]["role"] == "user"
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

    provider = OllamaProvider(ollama_url="http://127.0.0.1:11434", model="qwen3-vl:latest")
    text, usage = generate_answer(
        provider=provider,
        messages=[{"role": "user", "content": "Q"}],
    )

    assert text == "ollama rag answer"
    assert usage is not None
    assert usage["prompt_eval_count"] == 11
    assert usage["eval_count"] == 12
