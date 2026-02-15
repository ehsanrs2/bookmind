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


def test_generate_answer_ollama_response_fallback(monkeypatch) -> None:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "response": "ollama fallback answer",
                "prompt_eval_count": 7,
                "eval_count": 9,
            }

    def _fake_post(url, json=None, timeout=60):
        assert url == "http://127.0.0.1:11434/api/chat"
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

    provider = OllamaProvider(ollama_url="http://127.0.0.1:11434", model="qwen3-vl:latest")
    text, usage = generate_answer(
        provider=provider,
        messages=[{"role": "user", "content": "Q"}],
    )

    assert text == "ollama fallback answer"
    assert usage is not None
    assert usage["eval_count"] == 9


def test_generate_answer_ollama_malformed_raises(monkeypatch) -> None:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"unexpected": True}

    def _fake_post(url, json=None, timeout=60):
        assert url == "http://127.0.0.1:11434/api/chat"
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

    provider = OllamaProvider(ollama_url="http://127.0.0.1:11434", model="qwen3-vl:latest")
    try:
        generate_answer(
            provider=provider,
            messages=[{"role": "user", "content": "Q"}],
        )
        raise AssertionError("Expected RuntimeError for malformed Ollama response")
    except RuntimeError as exc:
        assert "failed after retries" in str(exc)


def test_generate_answer_ollama_retries_on_empty_then_succeeds(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    payloads = [
        {
            "message": {"content": "   \n"},
            "prompt_eval_count": 10,
            "eval_count": 512,
            "done_reason": "length",
        },
        {
            "response": "answer after retry [1:x]",
            "prompt_eval_count": 10,
            "eval_count": 12,
        },
    ]
    calls = {"count": 0}

    def _fake_post(url, json=None, timeout=60):
        assert url == "http://127.0.0.1:11434/api/chat"
        idx = calls["count"]
        calls["count"] += 1
        return _FakeResponse(payloads[idx])

    monkeypatch.setattr(requests, "post", _fake_post)
    attempts = []
    provider = OllamaProvider(ollama_url="http://127.0.0.1:11434", model="qwen3-vl:latest")
    text, usage = generate_answer(
        provider=provider,
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "Q"}],
        ollama_debug_hook=attempts.append,
    )

    assert text == "answer after retry [1:x]"
    assert usage is not None
    assert usage["attempt_index"] == 1
    assert calls["count"] == 2
    assert len(attempts) == 2
    assert attempts[0]["response"]["empty_answer"] is True


def test_generate_answer_ollama_retries_exhausted_raises(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self):
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"message": {"content": " \n\t"}, "eval_count": 512}

    calls = {"count": 0}

    def _fake_post(url, json=None, timeout=60):
        calls["count"] += 1
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)
    provider = OllamaProvider(ollama_url="http://127.0.0.1:11434", model="qwen3-vl:latest")
    try:
        generate_answer(
            provider=provider,
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "Q"}],
        )
        raise AssertionError("Expected RuntimeError after exhausted Ollama retries")
    except RuntimeError as exc:
        assert "failed after retries" in str(exc)
    assert calls["count"] == 3


def test_generate_answer_ollama_error_payload_retries_then_raises(monkeypatch) -> None:
    class _FakeResponse:
        status_code = 200
        headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"error": "model overloaded", "done_reason": "error"}

    def _fake_post(url, json=None, timeout=60):
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)
    attempts = []
    provider = OllamaProvider(ollama_url="http://127.0.0.1:11434", model="qwen3-vl:latest")
    try:
        generate_answer(
            provider=provider,
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "Q"}],
            ollama_debug_hook=attempts.append,
        )
        raise AssertionError("Expected RuntimeError for ollama error payload")
    except RuntimeError as exc:
        assert "failed after retries" in str(exc)
    assert len(attempts) == 3
    assert attempts[0]["response"]["provider_error"] == "model overloaded"
