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
                "message": {"content": '{"answer":"ollama rag answer","citations":[]}'},
                "prompt_eval_count": 11,
                "eval_count": 12,
            }

    def _fake_post(url, json=None, timeout=60):
        assert url == "http://127.0.0.1:11434/api/chat"
        assert json is not None
        assert json["messages"][0]["role"] == "user"
        assert json["think"] is False
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="text",
    )
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
                "response": '{"answer":"ollama fallback answer","citations":[]}',
                "prompt_eval_count": 7,
                "eval_count": 9,
            }

    def _fake_post(url, json=None, timeout=60):
        assert url == "http://127.0.0.1:11434/api/chat"
        assert json is not None
        assert json["think"] is False
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="text",
    )
    text, usage = generate_answer(
        provider=provider,
        messages=[{"role": "user", "content": "Q"}],
    )

    assert text == "ollama fallback answer"
    assert usage is not None
    assert usage["eval_count"] == 9


def test_generate_answer_ollama_json_content_extracts_answer(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self):
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "message": {
                    "content": '{"answer":"json answer","citations":[{"page":1,"stable_id":"s1"}]}'
                },
                "prompt_eval_count": 2,
                "eval_count": 3,
            }

    def _fake_post(url, json=None, timeout=60):
        assert json is not None
        assert "format" not in json
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="json",
    )
    text, usage = generate_answer(provider=provider, messages=[{"role": "user", "content": "Q"}])

    assert text == "json answer"
    assert usage is not None
    assert usage["attempt_index"] == 0


def test_generate_answer_ollama_malformed_json_raises(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self):
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"message": {"content": '{"answer": 123, "citations": []}'}}

    def _fake_post(url, json=None, timeout=60):
        assert url == "http://127.0.0.1:11434/api/chat"
        assert json is not None
        assert json["think"] is False
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

    attempts = []
    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="json",
    )
    try:
        generate_answer(
            provider=provider,
            messages=[{"role": "user", "content": "Q"}],
            ollama_debug_hook=attempts.append,
        )
        raise AssertionError("Expected RuntimeError for malformed Ollama JSON content")
    except RuntimeError as exc:
        assert "failed after retry" in str(exc)
    assert len(attempts) == 2
    assert attempts[0]["http_status"] == 200
    assert attempts[0]["response_headers"]["content-type"] == "application/json"


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
            "message": {"content": "   \n", "thinking": "draft answer"},
            "prompt_eval_count": 10,
            "eval_count": 512,
            "done_reason": "length",
        },
        {
            "message": {
                "content": '{"answer":"answer after retry [1:x]","citations":[{"page":1,"stable_id":"x"}]}'
            },
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
    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="json",
    )
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
    assert attempts[0]["response"]["empty_content_with_thinking"] is True
    assert attempts[0]["request"]["think"] is False
    assert attempts[0]["request"]["options"]["num_predict"] == 1536
    assert "format" not in attempts[0]["request"]
    assert attempts[1]["request"]["think"] is False
    assert attempts[1]["request"]["options"]["num_predict"] == 2048
    assert attempts[1]["request"]["options"]["temperature"] == 0.0


def test_generate_answer_ollama_uses_custom_num_predict_values(monkeypatch) -> None:
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
            "message": {"content": "", "thinking": "answer should be short"},
            "eval_count": 256,
            "done_reason": "length",
        },
        {
            "message": {"content": '{"answer":"ok","citations":[]}'},
            "eval_count": 42,
        },
    ]
    calls = {"count": 0}

    def _fake_post(url, json=None, timeout=60):
        idx = calls["count"]
        calls["count"] += 1
        return _FakeResponse(payloads[idx])

    monkeypatch.setattr(requests, "post", _fake_post)
    attempts = []
    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="text",
    )
    text, usage = generate_answer(
        provider=provider,
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "Q"}],
        ollama_num_predict=999,
        ollama_retry_num_predict=1337,
        ollama_debug_hook=attempts.append,
    )
    assert text == "ok"
    assert usage is not None
    assert calls["count"] == 2
    assert attempts[0]["request"]["think"] is False
    assert attempts[1]["request"]["think"] is False
    assert attempts[0]["request"]["options"]["num_predict"] == 999
    assert attempts[1]["request"]["options"]["num_predict"] == 1337


def test_generate_answer_ollama_accepts_string_think_level(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self):
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "message": {"content": '{"answer":"ok","citations":[]}'},
                "eval_count": 4,
                "prompt_eval_count": 3,
            }

    captured = {}

    def _fake_post(url, json=None, timeout=60):
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)
    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="text",
    )
    text, usage = generate_answer(
        provider=provider,
        messages=[{"role": "user", "content": "Q"}],
        ollama_think="high",
    )

    assert text == "ok"
    assert usage is not None
    assert captured["payload"]["think"] == "high"


def test_generate_answer_ollama_extractor_recovers_when_retry_empty_with_thinking(monkeypatch) -> None:
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
            "message": {"content": " \n\t", "thinking": "thinking attempt 1"},
            "eval_count": 512,
            "prompt_eval_count": 33,
            "done_reason": "length",
        },
        {
            "message": {"content": " \n\t", "thinking": "thinking attempt 2"},
            "eval_count": 512,
            "prompt_eval_count": 44,
            "done_reason": "length",
        },
        {
            "message": {
                "content": '{"answer":"final recovered answer","citations":[{"page":3,"stable_id":"sid-3"}]}'
            },
            "eval_count": 120,
            "prompt_eval_count": 21,
        },
    ]
    calls = {"count": 0}

    def _fake_post(url, json=None, timeout=60):
        idx = calls["count"]
        calls["count"] += 1
        return _FakeResponse(payloads[idx])

    monkeypatch.setattr(requests, "post", _fake_post)
    attempts = []
    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="json",
    )
    text, usage = generate_answer(
        provider=provider,
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "Q"}],
        ollama_think=True,
        ollama_debug_hook=attempts.append,
    )
    assert text == "final recovered answer"
    assert "thinking attempt 1" not in text
    assert "thinking attempt 2" not in text
    assert usage is not None
    assert usage["attempt_name"] == "extractor_json_only"
    assert calls["count"] == 3
    assert len(attempts) == 3
    assert attempts[0]["request"]["think"] is True
    assert attempts[1]["request"]["think"] is True
    assert attempts[2]["request"]["think"] is True
    assert attempts[0]["request"]["options"]["num_predict"] == 1536
    assert attempts[1]["request"]["options"]["num_predict"] == 2048
    assert attempts[2]["request"]["options"]["num_predict"] == 512
    assert attempts[2]["request"]["options"]["temperature"] == 0.0


def test_generate_answer_ollama_retries_exhausted_raises_without_thinking(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self):
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "message": {"content": " \n\t", "thinking": ""},
                "eval_count": 512,
                "prompt_eval_count": 33,
                "done_reason": "length",
            }

    calls = {"count": 0}

    def _fake_post(url, json=None, timeout=60):
        calls["count"] += 1
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)
    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="json",
    )
    try:
        generate_answer(
            provider=provider,
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "Q"}],
        )
        raise AssertionError("Expected RuntimeError after exhausted Ollama retries")
    except RuntimeError as exc:
        assert "empty assistant content with think=false" in str(exc)
        assert "done_reason=length" in str(exc)
        assert "eval_count=512" in str(exc)
        assert "prompt_eval_count=33" in str(exc)
        assert "raw_head_500=" in str(exc)
    assert calls["count"] == 2


def test_generate_answer_ollama_empty_content_with_thinking_raises_with_raw_excerpt(
    monkeypatch,
) -> None:
    class _FakeResponse:
        def __init__(self):
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "message": {"content": "", "thinking": "internal reasoning " * 60},
                "eval_count": 1536,
                "prompt_eval_count": 128,
                "done_reason": "length",
            }

    calls = {"count": 0}

    def _fake_post(url, json=None, timeout=60):
        assert json is not None
        assert json["think"] is False
        calls["count"] += 1
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)
    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="json",
    )
    try:
        generate_answer(
            provider=provider,
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "Q"}],
        )
        raise AssertionError("Expected RuntimeError for empty content with think=false")
    except RuntimeError as exc:
        err = str(exc)
        assert "empty assistant content with think=false" in err
        assert "done_reason=length" in err
        assert "eval_count=1536" in err
        assert "prompt_eval_count=128" in err
        assert "raw_head_500=" in err
    assert calls["count"] == 2


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
    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="json",
    )
    try:
        generate_answer(
            provider=provider,
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "Q"}],
            ollama_debug_hook=attempts.append,
        )
        raise AssertionError("Expected RuntimeError for ollama error payload")
    except RuntimeError as exc:
        assert "failed after retry" in str(exc)
    assert len(attempts) == 2
    assert attempts[0]["response"]["provider_error"] == "model overloaded"


def test_generate_answer_ollama_extracts_json_with_leading_trailing_junk(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self):
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "message": {
                    "content": (
                        "preface ignored\n"
                        '{"answer":"final answer","citations":[{"page":2,"stable_id":"s2"}]}'
                        "\ntrailer ignored"
                    )
                },
                "prompt_eval_count": 5,
                "eval_count": 6,
            }

    def _fake_post(url, json=None, timeout=60):
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

    provider = OllamaProvider(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        ollama_format="text",
    )
    text, usage = generate_answer(provider=provider, messages=[{"role": "user", "content": "Q"}])

    assert text == "final answer"
    assert usage is not None
    assert usage["attempt_index"] == 0
