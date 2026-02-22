from __future__ import annotations

import json
from pathlib import Path

from compare_baseline import build_compare_report, write_compare_report


def _report(
    *,
    reliability_score: float,
    hit_rate: float,
    mrr: float,
    verification_pass_rate: float,
    citations_resolvable_rate: float,
    meets_gate: bool,
) -> dict:
    return {
        "reliability_score": reliability_score,
        "meets_gate": meets_gate,
        "retrieval_summary": {
            "hit_rate@k": hit_rate,
            "mrr": mrr,
        },
        "generation_quality_summary": {
            "verification_pass_rate": verification_pass_rate,
            "citations_resolvable_rate": citations_resolvable_rate,
        },
    }


def test_build_compare_report_pass() -> None:
    baseline = _report(
        reliability_score=80.0,
        hit_rate=0.90,
        mrr=0.60,
        verification_pass_rate=0.92,
        citations_resolvable_rate=0.95,
        meets_gate=True,
    )
    current = _report(
        reliability_score=76.0,
        hit_rate=0.87,
        mrr=0.58,
        verification_pass_rate=0.88,
        citations_resolvable_rate=0.94,
        meets_gate=True,
    )
    report = build_compare_report(baseline_report=baseline, current_report=current)
    assert report["verdict"] == "PASS"
    assert report["deltas"]["reliability_score"] == -4.0
    assert report["deltas"]["hit_rate@k"] == -0.03
    assert report["deltas"]["mrr"] == -0.02
    assert report["deltas"]["verification_pass_rate"] == -0.04
    assert report["deltas"]["citations_resolvable_rate"] == -0.01
    assert report["regression_failures"] == []


def test_build_compare_report_fail_gate_precedence() -> None:
    baseline = _report(
        reliability_score=80.0,
        hit_rate=0.90,
        mrr=0.60,
        verification_pass_rate=0.92,
        citations_resolvable_rate=0.95,
        meets_gate=True,
    )
    current = _report(
        reliability_score=70.0,
        hit_rate=0.80,
        mrr=0.50,
        verification_pass_rate=0.80,
        citations_resolvable_rate=0.90,
        meets_gate=False,
    )
    report = build_compare_report(baseline_report=baseline, current_report=current)
    assert report["verdict"] == "FAIL_GATE"
    assert len(report["regression_failures"]) >= 1


def test_build_compare_report_fail_regression() -> None:
    baseline = _report(
        reliability_score=90.0,
        hit_rate=0.95,
        mrr=0.80,
        verification_pass_rate=0.96,
        citations_resolvable_rate=0.99,
        meets_gate=True,
    )
    current = _report(
        reliability_score=80.0,
        hit_rate=0.92,
        mrr=0.78,
        verification_pass_rate=0.93,
        citations_resolvable_rate=0.99,
        meets_gate=True,
    )
    report = build_compare_report(baseline_report=baseline, current_report=current)
    assert report["verdict"] == "FAIL_REGRESSION"
    assert report["regression_failures"][0]["metric"] == "reliability_score"


def test_write_compare_report_writes_json(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline_reliability_report.json"
    current_path = tmp_path / "current_reliability_report.json"
    out_dir = tmp_path / "compare"
    baseline_path.write_text(
        json.dumps(
            _report(
                reliability_score=80.0,
                hit_rate=0.90,
                mrr=0.60,
                verification_pass_rate=0.92,
                citations_resolvable_rate=0.95,
                meets_gate=True,
            )
        ),
        encoding="utf-8",
    )
    current_path.write_text(
        json.dumps(
            _report(
                reliability_score=80.0,
                hit_rate=0.90,
                mrr=0.60,
                verification_pass_rate=0.92,
                citations_resolvable_rate=0.95,
                meets_gate=True,
            )
        ),
        encoding="utf-8",
    )
    report = write_compare_report(
        baseline_report_path=baseline_path,
        current_report_path=current_path,
        output_dir=out_dir,
    )
    assert report["verdict"] == "PASS"
    assert (out_dir / "compare_report.json").exists()
    loaded = json.loads((out_dir / "compare_report.json").read_text(encoding="utf-8"))
    assert loaded["baseline"]["metrics"]["mrr"] == 0.6
