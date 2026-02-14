import pytest

import qdrant_ingest


def test_embed_offline_missing_model_has_clear_error(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    with pytest.raises(RuntimeError) as exc:
        qdrant_ingest._resolve_model_ref("org/nonexistent-offline-embed-model-for-test")

    message = str(exc.value)
    assert "offline mode" in message
    assert "Expected local model at:" in message
    assert "offline_bundle/models/embeddings" in message
