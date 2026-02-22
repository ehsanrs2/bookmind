"""Compare two reliability reports and produce a regression verdict."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

DEFAULT_REGRESSION_THRESHOLDS: Dict[str, float] = {
    "reliability_score_drop_max": 5.0,
    "hit_rate_at_k_drop_max": 0.05,
    "mrr_drop_max": 0.05,
    "verification_pass_rate_drop_max": 0.05,
}


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return 0.0


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _extract_metrics(report: Mapping[str, Any]) -> Dict[str, float]:
    retrieval_summary = report.get("retrieval_summary")
    if not isinstance(retrieval_summary, dict):
        retrieval_summary = {}
    generation_summary = report.get("generation_quality_summary")
    if not isinstance(generation_summary, dict):
        generation_summary = {}
    return {
        "reliability_score": round(_to_float(report.get("reliability_score")), 6),
        "hit_rate@k": round(_to_float(retrieval_summary.get("hit_rate@k")), 6),
        "mrr": round(_to_float(retrieval_summary.get("mrr")), 6),
        "verification_pass_rate": round(
            _to_float(generation_summary.get("verification_pass_rate")), 6
        ),
        "citations_resolvable_rate": round(
            _to_float(generation_summary.get("citations_resolvable_rate")), 6
        ),
    }


def _is_gate_pass(report: Mapping[str, Any]) -> bool:
    return bool(report.get("meets_gate"))


def build_compare_report(
    *,
    baseline_report: Mapping[str, Any],
    current_report: Mapping[str, Any],
    baseline_report_path: str | None = None,
    current_report_path: str | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    merged_thresholds = dict(DEFAULT_REGRESSION_THRESHOLDS)
    if thresholds:
        merged_thresholds.update({str(k): _to_float(v) for k, v in thresholds.items()})

    baseline_metrics = _extract_metrics(baseline_report)
    current_metrics = _extract_metrics(current_report)
    baseline_meets_gate = _is_gate_pass(baseline_report)
    current_meets_gate = _is_gate_pass(current_report)

    deltas: Dict[str, float] = {}
    for key, current_value in current_metrics.items():
        baseline_value = baseline_metrics.get(key, 0.0)
        deltas[key] = round(current_value - baseline_value, 6)

    regression_checks = [
        ("reliability_score", "reliability_score_drop_max"),
        ("hit_rate@k", "hit_rate_at_k_drop_max"),
        ("mrr", "mrr_drop_max"),
        ("verification_pass_rate", "verification_pass_rate_drop_max"),
    ]
    regression_failures = []
    for metric_name, threshold_name in regression_checks:
        baseline_value = baseline_metrics.get(metric_name, 0.0)
        current_value = current_metrics.get(metric_name, 0.0)
        drop = round(baseline_value - current_value, 6)
        threshold = round(_to_float(merged_thresholds.get(threshold_name)), 6)
        if drop > threshold:
            regression_failures.append(
                {
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "delta": round(current_value - baseline_value, 6),
                    "drop": drop,
                    "threshold": threshold,
                }
            )

    verdict = "PASS"
    if not current_meets_gate:
        verdict = "FAIL_GATE"
    elif regression_failures:
        verdict = "FAIL_REGRESSION"

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_report_path": baseline_report_path,
            "current_report_path": current_report_path,
        },
        "baseline": {
            "meets_gate": baseline_meets_gate,
            "metrics": baseline_metrics,
        },
        "current": {
            "meets_gate": current_meets_gate,
            "metrics": current_metrics,
        },
        "deltas": deltas,
        "regression_thresholds": merged_thresholds,
        "regression_failures": regression_failures,
        "verdict": verdict,
    }


def write_compare_report(
    *,
    baseline_report_path: str | Path,
    current_report_path: str | Path,
    output_dir: str | Path,
    thresholds: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    baseline_report = _load_json(baseline_report_path)
    current_report = _load_json(current_report_path)
    report = build_compare_report(
        baseline_report=baseline_report,
        current_report=current_report,
        baseline_report_path=str(baseline_report_path),
        current_report_path=str(current_report_path),
        thresholds=thresholds,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compare_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return report
