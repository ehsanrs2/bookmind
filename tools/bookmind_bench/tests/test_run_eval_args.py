from __future__ import annotations

from pathlib import Path

import run


def test_eval_args_defaults() -> None:
    parser = run.build_parser()
    args = parser.parse_args(["eval", "--queries", "queries.json", "--out", "eval_out"])
    assert args.qdrant_url == "http://127.0.0.1:6333"
    assert args.collection == "bookmind_bench"
    assert args.top_k == 8
    assert args.max_context_chars == 6000
    assert args.heuristic_boost_figures is True
    assert args.backend == "ollama"
    assert args.ollama_format == "json"
    assert args.ollama_num_ctx is None
    assert args.max_tokens == 512
    assert args.temperature == 0.2


def test_cmd_eval_parses_content_types_and_bool(tmp_path: Path, monkeypatch) -> None:
    queries_path = tmp_path / "queries.json"
    queries_path.write_text('[{"id":"q1","query":"x"}]\n', encoding="utf-8")

    captured = {}

    def _fake_run_eval(**kwargs):
        captured.update(kwargs)
        return {"summary": {"count_queries": 1, "avg_total_ms": 10, "avg_num_citations": 1}}

    monkeypatch.setattr(run, "run_eval", _fake_run_eval)
    monkeypatch.setattr(run, "write_reports", lambda results, output_dir: None)
    monkeypatch.setattr(
        run,
        "load_queries",
        lambda path: [{"id": "q1", "query": "diagram question"}],
    )

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--queries",
            str(queries_path),
            "--out",
            str(tmp_path / "eval"),
            "--content_types",
            "text,figure_caption",
            "--heuristic_boost_figures",
            "false",
            "--backend",
            "vllm",
        ]
    )
    exit_code = args.func(args)

    assert exit_code == 0
    assert captured["retrieval_cfg"]["content_types"] == ["text", "figure_caption"]
    assert captured["retrieval_cfg"]["heuristic_boost_figures"] is False
    assert captured["backend_cfg"]["backend"] == "vllm"
