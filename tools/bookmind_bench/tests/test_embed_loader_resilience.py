from __future__ import annotations

import sys
import types

import pytest

import qdrant_ingest


def test_load_embedding_model_retries_timeout_then_succeeds(monkeypatch) -> None:
    calls = {"count": 0}
    captured = {}

    class _FakeSentenceTransformer:
        def __init__(self, model_ref, **kwargs):
            calls["count"] += 1
            captured["kwargs"] = kwargs
            if calls["count"] == 1:
                raise RuntimeError("ReadTimeout while contacting huggingface")
            self.model_ref = model_ref

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(qdrant_ingest.time, "sleep", lambda _seconds: None)

    model = qdrant_ingest._load_embedding_model(
        "all-MiniLM-L6-v2",
        embed_cache_dir="/tmp/embed-cache",
        embed_local_only=False,
        hf_timeout_s=17,
        hf_retries=3,
    )

    assert model is not None
    assert calls["count"] == 2
    assert captured["kwargs"]["cache_folder"] == "/tmp/embed-cache"


def test_load_embedding_model_local_only_missing_fails_fast(monkeypatch) -> None:
    class _FakeSentenceTransformer:
        def __init__(self, model_ref, **kwargs):
            raise OSError("model files not found")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    with pytest.raises(RuntimeError) as exc:
        qdrant_ingest._load_embedding_model(
            "all-MiniLM-L6-v2",
            embed_local_only=True,
            hf_retries=5,
        )

    message = str(exc.value)
    assert "--embed_local_only=true" in message
    assert "Prefetch" in message
