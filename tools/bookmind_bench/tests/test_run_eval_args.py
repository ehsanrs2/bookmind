from __future__ import annotations

from pathlib import Path

import run


def test_eval_args_defaults() -> None:
    parser = run.build_parser()
    args = parser.parse_args(["eval", "--queries", "queries.json", "--out", "eval_out"])
    assert args.qdrant_url == "http://127.0.0.1:6333"
    assert args.collection == "bookmind_bench"
    assert args.embed_model == "all-MiniLM-L6-v2"
    assert args.embed_cache_dir is None
    assert args.embed_local_only is False
    assert args.hf_timeout_s == 30
    assert args.hf_retries == 3
    assert args.top_k == 8
    assert args.max_context_chars == 6000
    assert args.heuristic_boost_figures is True
    assert args.backend == "ollama"
    assert args.ollama_format == "text"
    assert args.ollama_num_ctx is None
    assert args.ollama_num_predict == 1536
    assert args.ollama_retry_num_predict == 2048
    assert args.max_tokens == 512
    assert args.temperature == 0.2
    assert args.progress is True


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
            "--embed_cache_dir",
            "/tmp/embed-cache",
            "--embed_local_only",
            "true",
            "--hf_timeout_s",
            "45",
            "--hf_retries",
            "7",
            "--backend",
            "vllm",
            "--ollama_num_predict",
            "1777",
            "--ollama_retry_num_predict",
            "2555",
            "--progress",
            "false",
        ]
    )
    exit_code = args.func(args)

    assert exit_code == 0
    assert captured["retrieval_cfg"]["content_types"] == ["text", "figure_caption"]
    assert captured["retrieval_cfg"]["heuristic_boost_figures"] is False
    assert captured["retrieval_cfg"]["embed_cache_dir"] == "/tmp/embed-cache"
    assert captured["retrieval_cfg"]["embed_local_only"] is True
    assert captured["retrieval_cfg"]["hf_timeout_s"] == 45
    assert captured["retrieval_cfg"]["hf_retries"] == 7
    assert captured["backend_cfg"]["backend"] == "vllm"
    assert captured["gen_cfg"]["ollama_num_predict"] == 1777
    assert captured["gen_cfg"]["ollama_retry_num_predict"] == 2555
    assert captured["progress"] is False


def test_eval_args_progress_toggle() -> None:
    parser = run.build_parser()
    args = parser.parse_args(
        ["eval", "--queries", "queries.json", "--out", "eval_out", "--progress", "false"]
    )
    assert args.progress is False
