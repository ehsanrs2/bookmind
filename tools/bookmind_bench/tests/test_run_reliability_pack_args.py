from __future__ import annotations

from pathlib import Path

import run


def test_reliability_pack_args_defaults() -> None:
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "reliability-pack",
            "--retrieval_report",
            "retrieval_report.json",
            "--eval_report",
            "report.json",
            "--out",
            "pack_out",
        ]
    )
    assert args.retrieval_report == "retrieval_report.json"
    assert args.eval_report == "report.json"
    assert args.out == "pack_out"


def test_cmd_reliability_pack_routes_calls(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def _fake_write_reliability_pack(**kwargs):
        captured.update(kwargs)
        return {"reliability_score": 88.0, "meets_gate": True, "recommendation": "PASS"}

    monkeypatch.setattr(run, "write_reliability_pack", _fake_write_reliability_pack)

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "reliability-pack",
            "--retrieval_report",
            str(tmp_path / "retrieval_report.json"),
            "--eval_report",
            str(tmp_path / "report.json"),
            "--out",
            str(tmp_path / "pack"),
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 0
    assert captured["retrieval_report_path"] == str(tmp_path / "retrieval_report.json")
    assert captured["eval_report_path"] == str(tmp_path / "report.json")
    assert captured["output_dir"] == tmp_path / "pack"
