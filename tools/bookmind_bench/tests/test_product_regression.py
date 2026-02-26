from __future__ import annotations

import json
from pathlib import Path

from product_regression import (
    build_product_compare_report,
    evaluate_check,
    load_product_checks,
    write_product_compare_report,
)


def test_load_product_checks_validates_schema(tmp_path: Path) -> None:
    checks_path = tmp_path / "checks.json"
    checks_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "id": "q1",
                        "query": "x",
                        "expected_status": "PASS",
                        "require_resolvable_citations": True,
                        "require_verification_pass": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    checks = load_product_checks(checks_path)
    assert checks[0]["id"] == "q1"
    assert checks[0]["expected_status"] == "PASS"


def test_evaluate_check_catches_safety_invariant() -> None:
    result = evaluate_check(
        check={
            "id": "s1",
            "query": "q",
            "expected_status": "FAIL_OK",
            "require_resolvable_citations": False,
            "require_verification_pass": False,
        },
        ask_result={
            "status": "PASS",
            "verification_status": "PASS",
            "citations_resolvable": False,
        },
    )
    assert result["ok"] is False
    assert "safety_pass_without_resolvable_citations" in result["blocking_failures"]


def _report(rows: list[dict], failed: int) -> dict:
    return {"summary": {"failed": failed}, "checks": rows}


def test_build_product_compare_report_detects_regression() -> None:
    baseline = _report(
        [
            {
                "id": "q_pass",
                "expected_status": "PASS",
                "ok": True,
                "actual_status": "PASS",
            }
        ],
        failed=0,
    )
    current = _report(
        [
            {
                "id": "q_pass",
                "expected_status": "PASS",
                "ok": False,
                "actual_status": "FAIL",
                "reasons": ["Expected status=PASS, got FAIL."],
            }
        ],
        failed=1,
    )
    report = build_product_compare_report(
        baseline_report=baseline,
        current_report=current,
        max_fail_increase=0,
    )
    assert report["verdict"] == "FAIL_REGRESSION"
    assert len(report["regressions"]) == 1


def test_write_product_compare_report_writes_file(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    out_dir = tmp_path / "out"
    baseline_path.write_text(
        json.dumps(
            _report(
                [
                    {
                        "id": "q_nf",
                        "expected_status": "NOT_FOUND",
                        "ok": True,
                        "actual_status": "NOT_FOUND",
                    }
                ],
                failed=0,
            )
        ),
        encoding="utf-8",
    )
    current_path.write_text(
        json.dumps(
            _report(
                [
                    {
                        "id": "q_nf",
                        "expected_status": "NOT_FOUND",
                        "ok": True,
                        "actual_status": "NOT_FOUND",
                    }
                ],
                failed=0,
            )
        ),
        encoding="utf-8",
    )
    report = write_product_compare_report(
        baseline_report_path=baseline_path,
        current_report_path=current_path,
        output_dir=out_dir,
    )
    assert report["verdict"] == "PASS"
    assert (out_dir / "compare_report.json").exists()
