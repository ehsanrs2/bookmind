from __future__ import annotations

import json
from pathlib import Path

import eval_pack


def test_run_eval_computes_summary_metrics(tmp_path: Path, monkeypatch) -> None:
    queries = [
        {"id": "q1", "query": "Explain the diagram", "expected_pages": [2]},
        {"id": "q2", "query": "List parts", "expected_pages": [9]},
    ]

    def _fake_search_query(**kwargs):
        query = kwargs["query"]
        if "diagram" in query.lower():
            return [
                {"stable_id": "t1", "page": 1, "content_type": "text", "text": "t", "meta": {}},
                {
                    "stable_id": "f1",
                    "page": 2,
                    "content_type": "figure_caption",
                    "text": "caption",
                    "meta": {},
                },
            ]
        return [
            {"stable_id": "t2", "page": 5, "content_type": "text", "text": "t", "meta": {}},
            {"stable_id": "t3", "page": 6, "content_type": "table", "text": "table", "meta": {}},
        ]

    class _FakeProvider:
        def chat(self, messages, max_tokens, temperature):
            user_text = messages[1]["content"]
            if "diagram" in user_text.lower():
                return "Wired path [2:f1]. No second cite.", {"total_tokens": 10}
            return "Parts listed. No citation marker here.", {"total_tokens": 12}

    class _Clock:
        def __init__(self):
            self.values = iter([0.00, 0.01, 0.01, 0.05, 1.00, 1.02, 1.02, 1.09])

        def __call__(self):
            return next(self.values)

    monkeypatch.setattr(eval_pack, "search_query", _fake_search_query)
    monkeypatch.setattr(eval_pack, "build_vlm_provider", lambda **kwargs: _FakeProvider())
    monkeypatch.setattr(eval_pack.time, "perf_counter", _Clock())

    results = eval_pack.run_eval(
        queries=queries,
        qdrant_url="http://127.0.0.1:6333",
        collection="bookmind_bench",
        backend_cfg={
            "backend": "ollama",
            "endpoint": "http://127.0.0.1:8000/v1",
            "model": "qwen3-vl",
            "ollama_url": "http://127.0.0.1:11434",
            "ollama_model": "qwen3-vl:latest",
        },
        retrieval_cfg={
            "top_k": 8,
            "content_types": ["text", "table", "figure_caption"],
            "max_context_chars": 6000,
            "heuristic_boost_figures": True,
            "embed_model": "all-MiniLM-L6-v2",
        },
        gen_cfg={"max_tokens": 512, "temperature": 0.2},
        output_dir=tmp_path,
    )

    summary = results["summary"]
    assert summary["count_queries"] == 2
    assert summary["avg_total_ms"] == 70.0
    assert summary["p50_total_ms"] == 70.0
    assert summary["avg_answer_len_chars"] > 0
    assert summary["avg_num_citations"] == 2.0
    assert summary["percent_queries_with_any_citation"] == 100.0
    assert summary["hit@k_pages"] == 50.0
    assert summary["expected_pages_evaluable_queries"] == 2

    q1 = results["queries"][0]
    assert q1["retrieved_ids"][0] == "f1"
    assert q1["citation_rate"] > 0
    assert q1["answer_text"]
    assert q1["answer_len_chars"] > 0

    q2 = results["queries"][1]
    assert q2["citation_rate"] == 0.0
    assert q2["answer_text"]
    assert q2["answer_len_chars"] > 0

    assert (tmp_path / "items" / "q1.json").exists()
    assert (tmp_path / "items" / "q2.json").exists()
    answer_q1_path = tmp_path / "per_query" / "q01_answer.txt"
    assert answer_q1_path.exists()
    assert answer_q1_path.read_text(encoding="utf-8").strip()
    assert (tmp_path / "per_query" / "q01_citations.json").exists()
    answer_q2_path = tmp_path / "per_query" / "q02_answer.txt"
    assert answer_q2_path.exists()
    assert answer_q2_path.read_text(encoding="utf-8").strip()
    assert (tmp_path / "per_query" / "q02_citations.json").exists()


def test_run_eval_writes_debug_artifact_for_empty_answer(tmp_path: Path, monkeypatch) -> None:
    queries = [{"id": "q1", "query": "Explain", "expected_pages": [1]}]

    def _fake_search_query(**kwargs):
        return [{"stable_id": "t1", "page": 1, "content_type": "text", "text": "ctx", "meta": {}}]

    class _FakeProvider:
        def chat(self, messages, max_tokens, temperature):
            return "   ", {"total_tokens": 5}

    monkeypatch.setattr(eval_pack, "search_query", _fake_search_query)
    monkeypatch.setattr(eval_pack, "build_vlm_provider", lambda **kwargs: _FakeProvider())

    results = eval_pack.run_eval(
        queries=queries,
        qdrant_url="http://127.0.0.1:6333",
        collection="bookmind_bench",
        backend_cfg={"backend": "ollama"},
        retrieval_cfg={},
        gen_cfg={},
        output_dir=tmp_path,
    )

    assert results["queries"][0]["answer_text"] == ""
    assert results["queries"][0]["status"] == "failed"
    answer_text_path = tmp_path / "per_query" / "q01_answer.txt"
    assert answer_text_path.exists()
    assert answer_text_path.read_text(encoding="utf-8").startswith("GENERATION_FAILED:")
    debug_path = tmp_path / "per_query" / "q01_raw_provider.json"
    assert debug_path.exists()
    debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert debug_payload["issue"] == "empty_answer"


def test_run_eval_generation_exception_includes_debug_artifact(tmp_path: Path, monkeypatch) -> None:
    queries = [{"id": "q1", "query": "Explain", "expected_pages": [1]}]

    def _fake_search_query(**kwargs):
        return [{"stable_id": "t1", "page": 1, "content_type": "text", "text": "ctx", "meta": {}}]

    class _FakeProvider:
        def chat(self, messages, max_tokens, temperature):
            raise ValueError("bad provider payload")

    monkeypatch.setattr(eval_pack, "search_query", _fake_search_query)
    monkeypatch.setattr(eval_pack, "build_vlm_provider", lambda **kwargs: _FakeProvider())

    results = eval_pack.run_eval(
        queries=queries,
        qdrant_url="http://127.0.0.1:6333",
        collection="bookmind_bench",
        backend_cfg={"backend": "ollama"},
        retrieval_cfg={},
        gen_cfg={},
        output_dir=tmp_path,
    )
    assert results["queries"][0]["status"] == "failed"
    assert results["queries"][0]["failure_reason"] == "bad provider payload"
    assert results["summary"]["failed_queries"] == 1

    debug_path = tmp_path / "per_query" / "q01_raw_provider.json"
    assert debug_path.exists()
    debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert debug_payload["issue"] == "generation_exception"
    assert debug_payload["error_type"] == "ValueError"

    answer_text_path = tmp_path / "per_query" / "q01_answer.txt"
    assert answer_text_path.exists()
    answer_line = answer_text_path.read_text(encoding="utf-8").strip()
    assert answer_line == "GENERATION_FAILED: bad provider payload"


def test_write_reports_report_json_includes_generation_cfg(tmp_path: Path) -> None:
    results = {
        "config": {
            "qdrant_url": "http://127.0.0.1:6333",
            "collection": "bookmind_bench",
            "backend": {"backend": "ollama"},
            "retrieval": {"top_k": 8},
            "generation": {
                "ollama_api": "generate",
                "ollama_num_predict": 2048,
                "ollama_retry_num_predict": 3072,
                "ollama_num_predict_auto": True,
                "max_context_chars": 7000,
            },
        },
        "summary": {"count_queries": 1, "failed_queries": 1},
        "queries": [
            {
                "id": "q1",
                "answer_text": "",
                "status": "failed",
                "failure_reason": "x",
                "timings": {"total_ms": 1},
                "citation_count": 0,
                "citation_rate": 0.0,
            }
        ],
    }
    eval_pack.write_reports(results, tmp_path)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    generation = report["config"]["generation"]
    assert generation["ollama_api"] == "generate"
    assert generation["ollama_num_predict"] == 2048
    assert generation["ollama_retry_num_predict"] == 3072
    assert report["summary"]["failed_queries"] == 1
    assert report["queries"][0]["status"] == "failed"
    assert report["queries"][0]["failure_reason"] == "x"


def test_run_eval_writes_ollama_attempt_artifacts_on_failure(tmp_path: Path, monkeypatch) -> None:
    queries = [{"id": "q1", "query": "Explain", "expected_pages": [1]}]

    def _fake_search_query(**kwargs):
        return [{"stable_id": "t1", "page": 1, "content_type": "text", "text": "ctx", "meta": {}}]

    class _FakeProvider:
        pass

    def _fake_generate_answer(
        provider,
        messages,
        max_tokens,
        temperature,
        ollama_num_predict=1536,
        ollama_retry_num_predict=2048,
        max_context_chars=6000,
        ollama_num_predict_auto=True,
        ollama_think=False,
        fallback_citations=None,
        ollama_debug_hook=None,
    ):
        if ollama_debug_hook is not None:
            ollama_debug_hook(
                {"attempt_name": "initial", "request": {"model": "m0"}, "response": {"answer_len_chars": 0}}
            )
            ollama_debug_hook(
                {
                    "attempt_name": "retry_strict_json",
                    "request": {"model": "m1"},
                    "response": {"answer_len_chars": 0},
                }
            )
            ollama_debug_hook(
                {
                    "attempt_name": "extractor_json_only",
                    "request": {"model": "m2"},
                    "response": {"answer_len_chars": 0},
                }
            )
        raise RuntimeError("Ollama generation failed after retries")

    monkeypatch.setattr(eval_pack, "search_query", _fake_search_query)
    monkeypatch.setattr(eval_pack, "build_vlm_provider", lambda **kwargs: _FakeProvider())
    monkeypatch.setattr(eval_pack, "generate_answer", _fake_generate_answer)

    results = eval_pack.run_eval(
        queries=queries,
        qdrant_url="http://127.0.0.1:6333",
        collection="bookmind_bench",
        backend_cfg={"backend": "ollama"},
        retrieval_cfg={},
        gen_cfg={},
        output_dir=tmp_path,
    )

    assert results["queries"][0]["status"] == "failed"
    assert (tmp_path / "per_query" / "q01_ollama_request.json").exists()
    assert (tmp_path / "per_query" / "q01_ollama_response.json").exists()
    assert (tmp_path / "per_query" / "q01_retry_request.json").exists()
    assert (tmp_path / "per_query" / "q01_retry_response.json").exists()
    assert (tmp_path / "per_query" / "q01_extract_request.json").exists()
    assert (tmp_path / "per_query" / "q01_extract_response.json").exists()
