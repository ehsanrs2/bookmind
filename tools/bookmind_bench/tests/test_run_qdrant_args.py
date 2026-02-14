import run


def test_qdrant_ingest_args_defaults():
    parser = run.build_parser()
    args = parser.parse_args(["qdrant-ingest", "--run", "run_dir"])
    assert args.bundle_dir is None
    assert args.qdrant_url == "http://127.0.0.1:6333"
    assert args.collection == "bookmind_bench"
    assert args.embed_model == "all-MiniLM-L6-v2"
    assert args.recreate is False
    assert args.batch_size == 64
    assert args.timeout_s == 60


def test_qdrant_search_content_types_parse(monkeypatch):
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "qdrant-search",
            "--query",
            "find table",
            "--content_types",
            "text,table",
        ]
    )

    captured = {}

    def _fake_search_query(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    exit_code = args.func(args)
    assert exit_code == 0
    assert captured["content_types"] == ["text", "table"]
