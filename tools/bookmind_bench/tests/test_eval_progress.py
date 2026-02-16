from __future__ import annotations

from pathlib import Path

import eval_pack


def _minimal_eval_run(tmp_path: Path, monkeypatch, *, progress: bool):
    queries = [{"id": "q1", "query": "Explain"}]

    def _fake_search_query(**kwargs):
        return [{"stable_id": "s1", "page": 1, "content_type": "text", "text": "ctx", "meta": {}}]

    class _FakeProvider:
        def chat(self, messages, max_tokens, temperature):
            return "answer [1:s1]", {"total_tokens": 1}

    monkeypatch.setattr(eval_pack, "search_query", _fake_search_query)
    monkeypatch.setattr(eval_pack, "build_vlm_provider", lambda **kwargs: _FakeProvider())
    return eval_pack.run_eval(
        queries=queries,
        qdrant_url="http://127.0.0.1:6333",
        collection="bookmind_bench",
        backend_cfg={"backend": "vllm"},
        retrieval_cfg={},
        gen_cfg={},
        output_dir=tmp_path,
        progress=progress,
    )


def test_eval_progress_fallback_prints_when_enabled(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(eval_pack, "_build_progress_reporter", lambda total, enabled: eval_pack._ProgressReporter(enabled=enabled, total=total, tqdm_bar=None))
    _minimal_eval_run(tmp_path, monkeypatch, progress=True)
    stdout = capsys.readouterr().out
    assert "[1/1] q1: retrieve..." in stdout
    assert "[1/1] q1: build_context..." in stdout
    assert "[1/1] q1: generate..." in stdout
    assert "[1/1] q1: write_artifacts..." in stdout


def test_eval_progress_silent_when_disabled(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(eval_pack, "_build_progress_reporter", lambda total, enabled: eval_pack._ProgressReporter(enabled=enabled, total=total, tqdm_bar=None))
    _minimal_eval_run(tmp_path, monkeypatch, progress=False)
    stdout = capsys.readouterr().out
    assert stdout == ""
