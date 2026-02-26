"""Product regression gate helpers for ask/check baseline workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_STATUSES = {"PASS", "NOT_FOUND", "FAIL_OK"}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def load_product_checks(path: str | Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("checks must be a non-empty array")

    validated: list[dict[str, Any]] = []
    for idx, item in enumerate(checks):
        if not isinstance(item, dict):
            raise ValueError(f"checks[{idx}] must be an object")
        check_id = str(item.get("id") or "").strip()
        query = str(item.get("query") or "").strip()
        expected_status = str(item.get("expected_status") or "").strip()
        if not check_id:
            raise ValueError(f"checks[{idx}].id is required")
        if not query:
            raise ValueError(f"checks[{idx}].query is required")
        if expected_status not in EXPECTED_STATUSES:
            raise ValueError(
                f"checks[{idx}].expected_status must be one of {sorted(EXPECTED_STATUSES)}"
            )
        validated.append(
            {
                "id": check_id,
                "query": query,
                "expected_status": expected_status,
                "require_resolvable_citations": bool(
                    item.get("require_resolvable_citations", False)
                ),
                "require_verification_pass": bool(
                    item.get("require_verification_pass", False)
                ),
            }
        )
    return validated


def evaluate_check(
    *,
    check: dict[str, Any],
    ask_result: dict[str, Any],
) -> dict[str, Any]:
    expected_status = str(check.get("expected_status") or "")
    actual_status = str(ask_result.get("status") or "")
    reasons: list[str] = []
    blocking_failures: list[str] = []

    if expected_status in {"PASS", "NOT_FOUND"} and actual_status != expected_status:
        message = f"Expected status={expected_status}, got {actual_status or 'UNKNOWN'}."
        reasons.append(message)
        blocking_failures.append("status_mismatch")

    require_resolvable = bool(check.get("require_resolvable_citations"))
    citations_resolvable = bool(ask_result.get("citations_resolvable"))
    if require_resolvable and not citations_resolvable:
        reasons.append("require_resolvable_citations=true but citations_resolvable=false.")
        blocking_failures.append("citations_unresolvable")

    require_verification = bool(check.get("require_verification_pass"))
    verification_status = str(ask_result.get("verification_status") or "")
    if require_verification and verification_status != "PASS":
        reasons.append(
            "require_verification_pass=true but verification_status is not PASS."
        )
        blocking_failures.append("verification_not_pass")

    # Safety invariant: FAIL_OK checks can pass, but only with resolvable citations.
    if expected_status == "FAIL_OK" and actual_status == "PASS" and not citations_resolvable:
        reasons.append(
            "FAIL_OK safety invariant violated: PASS output without citations_resolvable=true."
        )
        blocking_failures.append("safety_pass_without_resolvable_citations")

    return {
        "ok": len(blocking_failures) == 0,
        "reasons": reasons,
        "blocking_failures": blocking_failures,
    }


def build_product_compare_report(
    *,
    baseline_report: dict[str, Any],
    current_report: dict[str, Any],
    baseline_report_path: str | None = None,
    current_report_path: str | None = None,
    max_fail_increase: int | None = None,
) -> dict[str, Any]:
    baseline_checks = baseline_report.get("checks")
    current_checks = current_report.get("checks")
    if not isinstance(baseline_checks, list):
        baseline_checks = []
    if not isinstance(current_checks, list):
        current_checks = []

    baseline_by_id = {
        str(row.get("id") or ""): row
        for row in baseline_checks
        if isinstance(row, dict) and str(row.get("id") or "")
    }
    current_by_id = {
        str(row.get("id") or ""): row
        for row in current_checks
        if isinstance(row, dict) and str(row.get("id") or "")
    }

    regressions: list[dict[str, Any]] = []
    for check_id, baseline_row in baseline_by_id.items():
        expected_status = str(baseline_row.get("expected_status") or "")
        if expected_status not in {"PASS", "NOT_FOUND"}:
            continue
        baseline_ok = bool(baseline_row.get("ok"))
        current_row = current_by_id.get(check_id)
        current_ok = bool(current_row.get("ok")) if isinstance(current_row, dict) else False
        if baseline_ok and not current_ok:
            regressions.append(
                {
                    "id": check_id,
                    "expected_status": expected_status,
                    "baseline_ok": baseline_ok,
                    "current_ok": current_ok,
                    "baseline_status": str(baseline_row.get("actual_status") or ""),
                    "current_status": str((current_row or {}).get("actual_status") or ""),
                    "current_reasons": list((current_row or {}).get("reasons") or []),
                }
            )

    baseline_summary = (
        baseline_report.get("summary") if isinstance(baseline_report.get("summary"), dict) else {}
    )
    current_summary = (
        current_report.get("summary") if isinstance(current_report.get("summary"), dict) else {}
    )
    baseline_failed = int(baseline_summary.get("failed", 0) or 0)
    current_failed = int(current_summary.get("failed", 0) or 0)
    fail_increase = current_failed - baseline_failed

    threshold_failure: dict[str, Any] | None = None
    if max_fail_increase is not None and fail_increase > int(max_fail_increase):
        threshold_failure = {
            "metric": "failed_count",
            "baseline": baseline_failed,
            "current": current_failed,
            "increase": fail_increase,
            "threshold": int(max_fail_increase),
        }

    verdict = "PASS"
    if regressions:
        verdict = "FAIL_REGRESSION"
    elif threshold_failure is not None:
        verdict = "FAIL_THRESHOLD"

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_report_path": baseline_report_path,
            "current_report_path": current_report_path,
        },
        "baseline_summary": baseline_summary,
        "current_summary": current_summary,
        "regressions": regressions,
        "max_fail_increase": max_fail_increase,
        "threshold_failure": threshold_failure,
        "verdict": verdict,
    }


def write_product_compare_report(
    *,
    baseline_report_path: str | Path,
    current_report_path: str | Path,
    output_dir: str | Path,
    max_fail_increase: int | None = None,
) -> dict[str, Any]:
    baseline_report = _load_json(baseline_report_path)
    current_report = _load_json(current_report_path)
    report = build_product_compare_report(
        baseline_report=baseline_report,
        current_report=current_report,
        baseline_report_path=str(baseline_report_path),
        current_report_path=str(current_report_path),
        max_fail_increase=max_fail_increase,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compare_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return report
