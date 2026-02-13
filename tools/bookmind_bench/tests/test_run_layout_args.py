import run


def test_layout_args_accept_threshold_and_max_regions():
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "layout",
            "--imgdir",
            "images",
            "--out",
            "out_dir",
            "--score_thresh",
            "0.65",
            "--max_regions_per_page",
            "42",
        ]
    )
    assert args.score_thresh == 0.65
    assert args.max_regions_per_page == 42
