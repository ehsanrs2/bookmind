import run


def test_vlm_args_accept_max_tokens_and_temperature():
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "vlm",
            "--jobs",
            "jobs.jsonl",
            "--out",
            "out_dir",
            "--max_tokens",
            "123",
            "--temperature",
            "0.4",
        ]
    )
    assert args.max_tokens == 123
    assert args.temperature == 0.4


def test_vlm_args_parse_backend_and_ollama_defaults():
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "vlm",
            "--jobs",
            "jobs.jsonl",
            "--out",
            "out_dir",
            "--backend",
            "ollama",
        ]
    )
    assert args.backend == "ollama"
    assert args.ollama_url == "http://127.0.0.1:11434"
    assert args.ollama_model == "qwen3-vl:latest"


def test_cmd_vlm_routes_backend_to_engine(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("{}", encoding="utf-8")

    captured = {}

    def _fake_run_vlm_caption_jobs(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(run, "run_vlm_caption_jobs", _fake_run_vlm_caption_jobs)

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "vlm",
            "--jobs",
            str(jobs),
            "--out",
            str(tmp_path / "out"),
            "--backend",
            "ollama",
            "--ollama_url",
            "http://127.0.0.1:11434",
            "--ollama_model",
            "qwen3-vl:latest",
            "--endpoint",
            "http://ignored:8000/v1",
            "--model",
            "ignored-model",
        ]
    )

    exit_code = args.func(args)
    assert exit_code == 0
    assert captured["backend"] == "ollama"
    assert captured["ollama_url"] == "http://127.0.0.1:11434"
    assert captured["ollama_model"] == "qwen3-vl:latest"


def test_cmd_vlm_layout_routes_backend_to_engine(tmp_path, monkeypatch):
    imgdir = tmp_path / "images"
    imgdir.mkdir()
    layout_json = tmp_path / "layout.jsonl"
    layout_json.write_text("{}", encoding="utf-8")

    captured = {}

    def _fake_run_vlm_layout(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(run, "run_vlm_layout", _fake_run_vlm_layout)

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "vlm-layout",
            "--layout_json",
            str(layout_json),
            "--imgdir",
            str(imgdir),
            "--out",
            str(tmp_path / "out"),
            "--backend",
            "vllm",
            "--endpoint",
            "http://127.0.0.1:8000/v1",
            "--model",
            "qwen3-vl",
        ]
    )

    exit_code = args.func(args)
    assert exit_code == 0
    assert captured["backend"] == "vllm"
    assert captured["endpoint"] == "http://127.0.0.1:8000/v1"
    assert captured["model"] == "qwen3-vl"
