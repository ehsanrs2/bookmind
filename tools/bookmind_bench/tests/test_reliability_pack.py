from __future__ import annotations

import json
from pathlib import Path

from reliability_pack import build_reliability_report, write_reliability_pack


def test_build_reliability_report_score_and_gate_pass() -> None:
    retrieval_report = {
        "summary": {
            "hit_rate@k": 0.9,
            "mrr": 0.5,
            "avg_page_recall@k": 0.88,
        },
        "stability": {"avg_jaccard_topk": 0.92, "avg_jaccard_top5": 0.96},
    }
    eval_report = {
        "summary": {
            "avg_context_citations_per_query": 4.0,
            "avg_model_citations_per_query": 1.7,
            "verification_pass_rate": 0.9,
        },
        "queries": [
            {"citations_resolvable": True, "verification_status": "PASS"},
            {"citations_resolvable": True, "verification_status": "PASS"},
            {"citations_resolvable": True, "verification_status": "PASS"},
            {"citations_resolvable": True, "verification_status": "PASS"},
            {"citations_resolvable": True, "verification_status": "PASS"},
            {"citations_resolvable": True, "verification_status": "PASS"},
            {"citations_resolvable": True, "verification_status": "PASS"},
            {"citations_resolvable": True, "verification_status": "PASS"},
            {"citations_resolvable": True, "verification_status": "PASS"},
            {"citations_resolvable": False, "verification_status": "FAIL_UNSUPPORTED_CLAIMS"},
        ],
    }

    report = build_reliability_report(
        retrieval_report=retrieval_report,
        eval_report=eval_report,
        retrieval_report_path="/tmp/r.json",
        eval_report_path="/tmp/e.json",
    )

    assert report["reliability_score"] == 81.0
    assert report["meets_gate"] is True
    assert report["recommendation"] == "PASS"
    assert report["retrieval_summary"]["stability"]["avg_jaccard_top5"] == 0.96
    assert report["generation_quality_summary"]["citations_resolvable_rate"] == 0.9
    assert report["generation_quality_summary"]["avg_model_citations_per_answer"] == 1.7
    assert report["generation_quality_summary"]["avg_context_citations_per_answer"] == 4.0
    assert report["generation_quality_summary"]["avg_citations_per_answer"] == 1.7
    assert report["generation_quality_summary"]["verification_status_counts"]["PASS"] == 9


def test_build_reliability_report_recommendation_retrieval() -> None:
    report = build_reliability_report(
        retrieval_report={"summary": {"hit_rate@k": 0.8, "mrr": 0.5, "avg_page_recall@k": 0.7}},
        eval_report={
            "summary": {"avg_model_citations_per_query": 2.0, "verification_pass_rate": 0.95},
            "queries": [{"citations_resolvable": True, "verification_status": "PASS"}] * 10,
        },
    )
    assert report["meets_gate"] is False
    assert report["recommendation"] == "INVESTIGATE_RETRIEVAL"


def test_build_reliability_report_recommendation_citations() -> None:
    report = build_reliability_report(
        retrieval_report={"summary": {"hit_rate@k": 0.95, "mrr": 0.5, "avg_page_recall@k": 0.9}},
        eval_report={
            "summary": {"avg_model_citations_per_query": 2.0, "verification_pass_rate": 0.95},
            "queries": [{"citations_resolvable": False, "verification_status": "PASS"}] * 10,
        },
    )
    assert report["meets_gate"] is False
    assert report["recommendation"] == "INVESTIGATE_CITATIONS"


def test_build_reliability_report_recommendation_verifier() -> None:
    report = build_reliability_report(
        retrieval_report={"summary": {"hit_rate@k": 0.95, "mrr": 0.5, "avg_page_recall@k": 0.9}},
        eval_report={
            "summary": {"avg_model_citations_per_query": 2.0, "verification_pass_rate": 0.6},
            "queries": [{"citations_resolvable": True, "verification_status": "FAIL_UNSUPPORTED_CLAIMS"}]
            * 10,
        },
    )
    assert report["meets_gate"] is False
    assert report["recommendation"] == "INVESTIGATE_VERIFIER"


def test_write_reliability_pack_writes_json(tmp_path: Path) -> None:
    retrieval_path = tmp_path / "retrieval_report.json"
    eval_path = tmp_path / "report.json"
    out_dir = tmp_path / "pack"

    retrieval_path.write_text(
        json.dumps({"summary": {"hit_rate@k": 1.0, "mrr": 0.8, "avg_page_recall@k": 1.0}}),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "summary": {
                    "avg_context_citations_per_query": 3.0,
                    "avg_model_citations_per_query": 1.0,
                    "verification_pass_rate": 1.0,
                },
                "queries": [{"citations_resolvable": True, "verification_status": "PASS"}],
            }
        ),
        encoding="utf-8",
    )

    report = write_reliability_pack(
        retrieval_report_path=retrieval_path,
        eval_report_path=eval_path,
        output_dir=out_dir,
    )

    assert (out_dir / "reliability_report.json").exists()
    loaded = json.loads((out_dir / "reliability_report.json").read_text(encoding="utf-8"))
    assert loaded["recommendation"] == "PASS"
    assert report["meets_gate"] is True
