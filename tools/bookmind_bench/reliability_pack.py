"""Compose retrieval + eval artifacts into one decision-ready reliability report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "hit_rate_at_20_min": 0.85,
    "mrr_min": 0.35,
    "verification_pass_rate_min": 0.80,
    "citations_resolvable_rate_min": 0.90,
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "retrieval": 0.45,
    "verification": 0.35,
    "citations": 0.20,
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


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _verification_counts(queries: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in queries:
        if not isinstance(row, dict):
            continue
        key = str(row.get("verification_status") or "UNKNOWN").strip() or "UNKNOWN"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _citations_resolvable_rate(queries: Sequence[Dict[str, Any]]) -> float:
    if not queries:
        return 0.0
    resolvable = 0
    for row in queries:
        if not isinstance(row, dict):
            continue
        if bool(row.get("citations_resolvable")):
            resolvable += 1
    return _safe_rate(resolvable, len(queries))


def _verification_pass_rate(eval_report: Mapping[str, Any], queries: Sequence[Dict[str, Any]]) -> float:
    summary = eval_report.get("summary")
    if isinstance(summary, dict) and "verification_pass_rate" in summary:
        return round(_to_float(summary.get("verification_pass_rate")), 6)
    if not queries:
        return 0.0
    passes = 0
    for row in queries:
        if not isinstance(row, dict):
            continue
        if str(row.get("verification_status") or "") == "PASS":
            passes += 1
    return _safe_rate(passes, len(queries))


def _recommendation(
    *,
    meets_gate: bool,
    hit_rate: float,
    mrr: float,
    verification_pass_rate: float,
    citations_resolvable_rate: float,
    thresholds: Mapping[str, float],
) -> str:
    if meets_gate:
        return "PASS"
    if hit_rate < _to_float(thresholds.get("hit_rate_at_20_min")) or mrr < _to_float(
        thresholds.get("mrr_min")
    ):
        return "INVESTIGATE_RETRIEVAL"
    if citations_resolvable_rate < _to_float(thresholds.get("citations_resolvable_rate_min")):
        return "INVESTIGATE_CITATIONS"
    if verification_pass_rate < _to_float(thresholds.get("verification_pass_rate_min")):
        return "INVESTIGATE_VERIFIER"
    return "INVESTIGATE_VERIFIER"


def build_reliability_report(
    *,
    retrieval_report: Mapping[str, Any],
    eval_report: Mapping[str, Any],
    retrieval_report_path: str | None = None,
    eval_report_path: str | None = None,
    thresholds: Mapping[str, float] | None = None,
    weights: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    merged_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        merged_thresholds.update({str(k): _to_float(v) for k, v in thresholds.items()})

    merged_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        merged_weights.update({str(k): _to_float(v) for k, v in weights.items()})

    retrieval_summary = retrieval_report.get("summary")
    if not isinstance(retrieval_summary, dict):
        retrieval_summary = {}
    retrieval_stability = retrieval_report.get("stability")
    if not isinstance(retrieval_stability, dict):
        retrieval_stability = None

    eval_summary = eval_report.get("summary")
    if not isinstance(eval_summary, dict):
        eval_summary = {}
    eval_queries = eval_report.get("queries")
    if not isinstance(eval_queries, list):
        eval_queries = []

    hit_rate = round(_to_float(retrieval_summary.get("hit_rate@k")), 6)
    mrr = round(_to_float(retrieval_summary.get("mrr")), 6)
    avg_page_recall = round(_to_float(retrieval_summary.get("avg_page_recall@k")), 6)

    avg_model_citations = round(
        _to_float(
            eval_summary.get(
                "avg_model_citations_per_query",
                eval_summary.get("avg_num_citations"),
            )
        ),
        6,
    )
    avg_context_citations = round(
        _to_float(
            eval_summary.get(
                "avg_context_citations_per_query",
                eval_summary.get("avg_num_citations"),
            )
        ),
        6,
    )
    # citations_resolvable_rate is intentionally based on model citation parse+resolution fields.
    citations_resolvable_rate = _citations_resolvable_rate(eval_queries)
    verification_pass_rate = _verification_pass_rate(eval_report, eval_queries)
    verification_counts = _verification_counts(eval_queries)

    retrieval_component = (hit_rate + mrr) / 2.0
    reliability_score = round(
        100.0
        * (
            _to_float(merged_weights.get("retrieval")) * retrieval_component
            + _to_float(merged_weights.get("verification")) * verification_pass_rate
            + _to_float(merged_weights.get("citations")) * citations_resolvable_rate
        ),
        3,
    )

    meets_gate = (
        hit_rate >= _to_float(merged_thresholds.get("hit_rate_at_20_min"))
        and mrr >= _to_float(merged_thresholds.get("mrr_min"))
        and verification_pass_rate >= _to_float(merged_thresholds.get("verification_pass_rate_min"))
        and citations_resolvable_rate
        >= _to_float(merged_thresholds.get("citations_resolvable_rate_min"))
    )

    recommendation = _recommendation(
        meets_gate=bool(meets_gate),
        hit_rate=hit_rate,
        mrr=mrr,
        verification_pass_rate=verification_pass_rate,
        citations_resolvable_rate=citations_resolvable_rate,
        thresholds=merged_thresholds,
    )

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "retrieval_report_path": retrieval_report_path,
            "eval_report_path": eval_report_path,
        },
        "retrieval_summary": {
            "hit_rate@k": hit_rate,
            "mrr": mrr,
            "avg_page_recall@k": avg_page_recall,
            "stability": retrieval_stability,
        },
        "generation_quality_summary": {
            "citations_resolvable_rate": citations_resolvable_rate,
            "avg_model_citations_per_answer": avg_model_citations,
            "avg_context_citations_per_answer": avg_context_citations,
            # Backward-compatible alias.
            "avg_citations_per_answer": avg_model_citations,
            "verification_pass_rate": verification_pass_rate,
            "verification_status_counts": verification_counts,
        },
        "score_components": {
            "weights": merged_weights,
            "retrieval_component": round(retrieval_component, 6),
            "verification_component": verification_pass_rate,
            "citations_component": citations_resolvable_rate,
        },
        "thresholds": merged_thresholds,
        "reliability_score": reliability_score,
        "meets_gate": bool(meets_gate),
        "recommendation": recommendation,
    }


def write_reliability_pack(
    *,
    retrieval_report_path: str | Path,
    eval_report_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    retrieval_report = _load_json(retrieval_report_path)
    eval_report = _load_json(eval_report_path)
    report = build_reliability_report(
        retrieval_report=retrieval_report,
        eval_report=eval_report,
        retrieval_report_path=str(retrieval_report_path),
        eval_report_path=str(eval_report_path),
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reliability_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return report
