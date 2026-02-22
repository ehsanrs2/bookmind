from __future__ import annotations

from pathlib import Path

import run


def test_retrieval_eval_args_defaults() -> None:
    parser = run.build_parser()
    args = parser.parse_args(["retrieval-eval", "--queries", "queries.json", "--out", "retrieval_out"])
    assert args.qdrant_url == "http://127.0.0.1:6333"
    assert args.collection == "bookmind_bench"
    assert args.top_k == 20
    assert args.content_types == "text,figure_caption"
    assert args.progress is True
    assert args.repeat == 1
    assert args.seed is None


def test_cmd_retrieval_eval_routes_calls(tmp_path: Path, monkeypatch) -> None:
    queries_path = tmp_path / "queries.json"
    queries_path.write_text('[{"id":"q1","query":"x","truth_pages":[1]}]\n', encoding="utf-8")

    captured = {}

    monkeypatch.setattr(run, "load_queries", lambda path: [{"id": "q1", "query": "x", "truth_pages": [1]}])

    def _fake_run_retrieval_eval(**kwargs):
        captured.update(kwargs)
        return {
            "summary": {
                "count_queries": 1,
                "hit_rate@k": 1.0,
                "avg_page_recall@k": 1.0,
                "mrr": 1.0,
            }
        }

    monkeypatch.setattr(run, "run_retrieval_eval", _fake_run_retrieval_eval)

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "retrieval-eval",
            "--queries",
            str(queries_path),
            "--out",
            str(tmp_path / "out"),
            "--content_types",
            "figure_caption",
            "--repeat",
            "3",
            "--progress",
            "false",
            "--seed",
            "7",
        ]
    )
    exit_code = args.func(args)

    assert exit_code == 0
    assert captured["top_k"] == 20
    assert captured["content_types"] == ["figure_caption"]
    assert captured["repeat"] == 3
    assert captured["progress"] is False
    assert captured["seed"] == 7
    assert captured["queries_path"] == str(queries_path)
