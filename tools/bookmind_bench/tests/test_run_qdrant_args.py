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
    assert args.id_mode == "uint64"


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


def test_qdrant_ingest_auto_bundle_when_bundle_missing(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    ingest_dir = run_dir / "ingest"
    ingest_dir.mkdir(parents=True)
    (ingest_dir / "records.jsonl").write_text("{}\n", encoding="utf-8")

    captured = {}

    class _FakeBundleResult:
        bundle_dir = str(run_dir / "bundle")

    def _fake_bundle_run(**kwargs):
        captured["bundle_run"] = kwargs
        (run_dir / "bundle").mkdir(parents=True, exist_ok=True)
        return _FakeBundleResult()

    def _fake_ingest_bundle(**kwargs):
        captured["ingest_bundle"] = kwargs
        return {
            "collection": "bookmind_bench",
            "records_ingested": 1,
            "records_total": 1,
            "records_failed": 0,
            "records_skipped_missing_id": 0,
            "records_skipped_invalid_id": 0,
            "collection_recreated": False,
            "collection_created": True,
            "id_mode": "uint64",
        }

    monkeypatch.setattr(run, "bundle_run", _fake_bundle_run)
    monkeypatch.setattr(run, "ingest_bundle", _fake_ingest_bundle)

    parser = run.build_parser()
    args = parser.parse_args(["qdrant-ingest", "--run", str(run_dir)])
    exit_code = args.func(args)

    assert exit_code == 0
    assert captured["bundle_run"]["run_dir"] == str(run_dir)
    assert captured["bundle_run"]["include_images"] is False
    assert captured["ingest_bundle"]["bundle_dir"] == str(run_dir / "bundle")
