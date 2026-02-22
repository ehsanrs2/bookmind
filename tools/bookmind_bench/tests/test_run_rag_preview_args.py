import run


def test_rag_preview_args_defaults() -> None:
    parser = run.build_parser()
    args = parser.parse_args(["rag-preview", "--query", "q"])
    assert args.qdrant_url == "http://127.0.0.1:6333"
    assert args.collection == "bookmind_bench"
    assert args.embed_model == "all-MiniLM-L6-v2"
    assert args.embed_cache_dir is None
    assert args.embed_local_only is False
    assert args.hf_timeout_s == 30
    assert args.hf_retries == 3
    assert args.top_k == 8
    assert args.max_context_chars == 6000
    assert args.backend == "ollama"
    assert args.ollama_url == "http://127.0.0.1:11434"
    assert args.ollama_model == "qwen3-vl:latest"
    assert args.ollama_format == "text"
    assert args.ollama_num_ctx is None
    assert args.ollama_api == "generate"
    assert args.ollama_num_predict == 1536
    assert args.ollama_retry_num_predict == 2048
    assert args.ollama_num_predict_auto is True
    assert args.ollama_think is False
    assert args.max_tokens == 512
    assert args.temperature == 0.2
    assert args.show_snippets is False
    assert args.require_resolvable_citations is False
    assert args.enforce_verified is False
    assert args.force_citations is False
    assert args.citation_repair_retry is None
    assert args.citation_min_count == 1


def test_cmd_rag_preview_routes_calls(tmp_path, monkeypatch) -> None:
    captured = {}

    def _fake_search_query(**kwargs):
        captured["search"] = kwargs
        return [
            {
                "stable_id": "s1",
                "page": 1,
                "bbox": [0, 0, 1, 1],
                "content_type": "text",
                "text": "hello",
                "meta": {},
            }
        ]

    class _FakeProvider:
        pass

    def _fake_build_provider(**kwargs):
        captured["provider"] = kwargs
        return _FakeProvider()

    def _fake_generate_answer(**kwargs):
        captured["generate_answer"] = kwargs
        return "hello [1:s1]", {"total_tokens": 1}

    def _fake_format_preview_output(answer_text, citations, show_snippets, model_citations=None):
        assert answer_text == "hello [1:s1]"
        assert isinstance(model_citations, dict)
        return "ok\n"

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    monkeypatch.setattr(run, "build_vlm_provider", _fake_build_provider)
    monkeypatch.setattr(run, "generate_answer", _fake_generate_answer)
    monkeypatch.setattr(run, "format_preview_output", _fake_format_preview_output)

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "rag-preview",
            "--query",
            "find x",
            "--content_types",
            "text,table",
            "--max_tokens",
            "123",
            "--temperature",
            "0.4",
            "--embed_cache_dir",
            "/tmp/embed-cache",
            "--embed_local_only",
            "true",
            "--hf_timeout_s",
            "41",
            "--hf_retries",
            "6",
            "--ollama_num_predict",
            "1666",
            "--ollama_retry_num_predict",
            "2444",
            "--ollama_think",
            "medium",
            "--ollama_api",
            "chat",
        ]
    )

    exit_code = args.func(args)
    assert exit_code == 0
    assert captured["search"]["content_types"] == ["text", "table"]
    assert captured["search"]["embed_cache_dir"] == "/tmp/embed-cache"
    assert captured["search"]["embed_local_only"] is True
    assert captured["search"]["hf_timeout_s"] == 41
    assert captured["search"]["hf_retries"] == 6
    assert captured["provider"]["backend"] == "ollama"
    assert captured["provider"]["ollama_api"] == "chat"
    assert captured["generate_answer"]["max_tokens"] == 123
    assert captured["generate_answer"]["temperature"] == 0.4
    assert captured["generate_answer"]["ollama_num_predict"] == 1666
    assert captured["generate_answer"]["ollama_retry_num_predict"] == 2444
    assert captured["generate_answer"]["ollama_num_predict_auto"] is True
    assert captured["generate_answer"]["max_context_chars"] == 6000
    assert captured["generate_answer"]["ollama_think"] == "medium"


def test_cmd_rag_preview_autotunes_large_context_num_predict(monkeypatch) -> None:
    captured = {}

    def _fake_search_query(**kwargs):
        return [{"stable_id": "s1", "page": 1, "content_type": "text", "text": "x", "meta": {}}]

    class _FakeProvider:
        pass

    def _fake_build_provider(**kwargs):
        return _FakeProvider()

    def _fake_generate_answer(**kwargs):
        captured.update(kwargs)
        return "answer", {"total_tokens": 1}

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    monkeypatch.setattr(run, "build_vlm_provider", _fake_build_provider)
    monkeypatch.setattr(run, "generate_answer", _fake_generate_answer)
    monkeypatch.setattr(run, "format_preview_output", lambda **kwargs: "ok\n")

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "rag-preview",
            "--query",
            "find x",
            "--max_context_chars",
            "7000",
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 0
    assert captured["ollama_num_predict"] == 2048
    assert captured["ollama_retry_num_predict"] == 3072


def test_cmd_rag_preview_can_disable_autotune(monkeypatch) -> None:
    captured = {}

    def _fake_search_query(**kwargs):
        return [{"stable_id": "s1", "page": 1, "content_type": "text", "text": "x", "meta": {}}]

    class _FakeProvider:
        pass

    def _fake_build_provider(**kwargs):
        return _FakeProvider()

    def _fake_generate_answer(**kwargs):
        captured.update(kwargs)
        return "answer", {"total_tokens": 1}

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    monkeypatch.setattr(run, "build_vlm_provider", _fake_build_provider)
    monkeypatch.setattr(run, "generate_answer", _fake_generate_answer)
    monkeypatch.setattr(run, "format_preview_output", lambda **kwargs: "ok\n")

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "rag-preview",
            "--query",
            "find x",
            "--max_context_chars",
            "9000",
            "--ollama_num_predict",
            "1700",
            "--ollama_retry_num_predict",
            "2400",
            "--ollama_num_predict_auto",
            "false",
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 0
    assert captured["ollama_num_predict"] == 1700
    assert captured["ollama_retry_num_predict"] == 2400
    assert captured["ollama_num_predict_auto"] is False


def test_cmd_rag_preview_require_resolvable_citations_fails_on_none(monkeypatch) -> None:
    def _fake_search_query(**kwargs):
        return [{"stable_id": "s1", "chunk_id": "v1_c1", "page": 1, "content_type": "text", "text": "x", "meta": {}}]

    class _FakeProvider:
        pass

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    monkeypatch.setattr(run, "build_vlm_provider", lambda **kwargs: _FakeProvider())
    monkeypatch.setattr(run, "generate_answer", lambda **kwargs: ("answer without cites", {"total_tokens": 1}))
    monkeypatch.setattr(run, "format_preview_output", lambda **kwargs: "ok\n")

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "rag-preview",
            "--query",
            "find x",
            "--require_resolvable_citations",
            "true",
        ]
    )
    try:
        args.func(args)
        raise AssertionError("Expected SystemExit")
    except SystemExit as exc:
        assert "NO_CITATIONS" in str(exc)


def test_cmd_rag_preview_force_citations_enables_default_repair_retry(monkeypatch) -> None:
    captured = {"calls": 0}

    def _fake_search_query(**kwargs):
        return [{"stable_id": "s1", "chunk_id": "v1_c1", "page": 1, "content_type": "text", "text": "ctx", "meta": {}}]

    class _FakeProvider:
        pass

    def _fake_generate_answer(**kwargs):
        captured["calls"] += 1
        if captured["calls"] == 1:
            return "answer without citations", {"total_tokens": 1}
        return "answer [1:s1]", {"total_tokens": 1}

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    monkeypatch.setattr(run, "build_vlm_provider", lambda **kwargs: _FakeProvider())
    monkeypatch.setattr(run, "generate_answer", _fake_generate_answer)
    monkeypatch.setattr(run, "format_preview_output", lambda **kwargs: "ok\n")

    parser = run.build_parser()
    args = parser.parse_args(["rag-preview", "--query", "find x", "--force_citations", "true"])
    exit_code = args.func(args)
    assert exit_code == 0
    assert captured["calls"] == 2


def test_cmd_rag_preview_require_resolvable_citations_fails_on_unresolved(monkeypatch) -> None:
    def _fake_search_query(**kwargs):
        return [{"stable_id": "s1", "chunk_id": "v1_c1", "page": 1, "content_type": "text", "text": "x", "meta": {}}]

    class _FakeProvider:
        pass

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    monkeypatch.setattr(run, "build_vlm_provider", lambda **kwargs: _FakeProvider())
    monkeypatch.setattr(run, "generate_answer", lambda **kwargs: ("answer [2:s2]", {"total_tokens": 1}))
    monkeypatch.setattr(run, "format_preview_output", lambda **kwargs: "ok\n")

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "rag-preview",
            "--query",
            "find x",
            "--require_resolvable_citations",
            "true",
        ]
    )
    try:
        args.func(args)
        raise AssertionError("Expected SystemExit")
    except SystemExit as exc:
        assert "UNRESOLVABLE_CITATIONS" in str(exc)


def test_cmd_rag_preview_enforce_verified_fails(monkeypatch) -> None:
    def _fake_search_query(**kwargs):
        return [{"stable_id": "s1", "chunk_id": "v1_c1", "page": 1, "content_type": "text", "text": "ctx", "meta": {}}]

    class _FakeProvider:
        pass

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    monkeypatch.setattr(run, "build_vlm_provider", lambda **kwargs: _FakeProvider())
    monkeypatch.setattr(run, "generate_answer", lambda **kwargs: ("answer without cites", {"total_tokens": 1}))
    monkeypatch.setattr(run, "format_preview_output", lambda **kwargs: "ok\n")

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "rag-preview",
            "--query",
            "find x",
            "--enforce_verified",
            "true",
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 2
