from __future__ import annotations

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
    assert (tmp_path / "per_query" / "q01_answer.txt").exists()
    assert (tmp_path / "per_query" / "q01_citations.json").exists()
    assert (tmp_path / "per_query" / "q02_answer.txt").exists()
    assert (tmp_path / "per_query" / "q02_citations.json").exists()
