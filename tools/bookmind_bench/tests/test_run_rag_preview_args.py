import run


def test_rag_preview_args_defaults() -> None:
    parser = run.build_parser()
    args = parser.parse_args(["rag-preview", "--query", "q"])
    assert args.qdrant_url == "http://127.0.0.1:6333"
    assert args.collection == "bookmind_bench"
    assert args.top_k == 8
    assert args.max_context_chars == 6000
    assert args.backend == "ollama"
    assert args.ollama_url == "http://127.0.0.1:11434"
    assert args.ollama_model == "qwen3-vl:latest"
    assert args.max_tokens == 512
    assert args.temperature == 0.2
    assert args.show_snippets is False


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
        def chat(self, messages, max_tokens, temperature):
            captured["chat"] = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            return "answer", {"total_tokens": 1}

    def _fake_build_provider(**kwargs):
        captured["provider"] = kwargs
        return _FakeProvider()

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    monkeypatch.setattr(run, "build_vlm_provider", _fake_build_provider)

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
        ]
    )

    exit_code = args.func(args)
    assert exit_code == 0
    assert captured["search"]["content_types"] == ["text", "table"]
    assert captured["provider"]["backend"] == "ollama"
    assert captured["chat"]["max_tokens"] == 123
    assert captured["chat"]["temperature"] == 0.4
