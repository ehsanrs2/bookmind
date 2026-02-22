from __future__ import annotations

from pathlib import Path

import run


def test_compare_baseline_args_defaults() -> None:
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "compare-baseline",
            "--baseline",
            "baseline_reliability_report.json",
            "--current",
            "current_reliability_report.json",
            "--out",
            "compare_out",
        ]
    )
    assert args.baseline == "baseline_reliability_report.json"
    assert args.current == "current_reliability_report.json"
    assert args.out == "compare_out"
    assert args.max_reliability_score_drop == 5.0
    assert args.max_verification_pass_rate_drop == 0.05
    assert args.max_hit_rate_drop == 0.05
    assert args.max_mrr_drop == 0.05


def test_cmd_compare_baseline_routes_calls(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def _fake_write_compare_report(**kwargs):
        captured.update(kwargs)
        return {"verdict": "PASS", "current": {"meets_gate": True}, "regression_failures": []}

    monkeypatch.setattr(run, "write_compare_report", _fake_write_compare_report)

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "compare-baseline",
            "--baseline",
            str(tmp_path / "baseline_reliability_report.json"),
            "--current",
            str(tmp_path / "current_reliability_report.json"),
            "--out",
            str(tmp_path / "compare"),
            "--max_reliability_score_drop",
            "7.5",
            "--max_verification_pass_rate_drop",
            "0.07",
            "--max_hit_rate_drop",
            "0.08",
            "--max_mrr_drop",
            "0.09",
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 0
    assert captured["baseline_report_path"] == str(tmp_path / "baseline_reliability_report.json")
    assert captured["current_report_path"] == str(tmp_path / "current_reliability_report.json")
    assert captured["output_dir"] == tmp_path / "compare"
    assert captured["thresholds"]["reliability_score_drop_max"] == 7.5
    assert captured["thresholds"]["verification_pass_rate_drop_max"] == 0.07
    assert captured["thresholds"]["hit_rate_at_k_drop_max"] == 0.08
    assert captured["thresholds"]["mrr_drop_max"] == 0.09
