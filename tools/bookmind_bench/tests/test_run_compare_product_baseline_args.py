from __future__ import annotations

from pathlib import Path

import run


def test_compare_product_baseline_args_defaults() -> None:
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "compare-product-baseline",
            "--baseline_report",
            "baseline_report.json",
            "--current_report",
            "current_report.json",
            "--out",
            "compare_out",
        ]
    )
    assert args.baseline_report == "baseline_report.json"
    assert args.current_report == "current_report.json"
    assert args.out == "compare_out"
    assert args.max_fail_increase is None


def test_cmd_compare_product_baseline_routes_calls(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def _fake_write_product_compare_report(**kwargs):
        captured.update(kwargs)
        return {"verdict": "PASS", "regressions": [], "threshold_failure": None}

    monkeypatch.setattr(run, "write_product_compare_report", _fake_write_product_compare_report)

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "compare-product-baseline",
            "--baseline_report",
            str(tmp_path / "baseline_report.json"),
            "--current_report",
            str(tmp_path / "current_report.json"),
            "--out",
            str(tmp_path / "compare"),
            "--max_fail_increase",
            "2",
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 0
    assert captured["baseline_report_path"] == str(tmp_path / "baseline_report.json")
    assert captured["current_report_path"] == str(tmp_path / "current_report.json")
    assert captured["output_dir"] == tmp_path / "compare"
    assert captured["max_fail_increase"] == 2


def test_cmd_compare_product_baseline_nonzero_on_fail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        run,
        "write_product_compare_report",
        lambda **kwargs: {
            "verdict": "FAIL_REGRESSION",
            "regressions": [{"id": "q1"}],
            "threshold_failure": None,
        },
    )
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "compare-product-baseline",
            "--baseline_report",
            str(tmp_path / "baseline_report.json"),
            "--current_report",
            str(tmp_path / "current_report.json"),
            "--out",
            str(tmp_path / "compare"),
        ]
    )
    assert args.func(args) == 2
