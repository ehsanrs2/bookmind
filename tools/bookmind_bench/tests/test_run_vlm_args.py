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
