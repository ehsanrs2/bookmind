from __future__ import annotations

import json
from pathlib import Path

import run


def _write_checks(path: Path, checks: list[dict]) -> None:
    path.write_text(json.dumps({"checks": checks}) + "\n", encoding="utf-8")


def test_product_check_args_defaults() -> None:
    parser = run.build_parser()
    args = parser.parse_args(
        [
            "product-check",
            "--checks",
            "checks.json",
            "--out",
            "product_out",
        ]
    )
    assert args.checks == "checks.json"
    assert args.out == "product_out"
    assert args.qdrant_url == "http://127.0.0.1:6333"
    assert args.collection == "bookmind_bench"
    assert args.top_k == 20
    assert args.content_types == "text,figure_caption"
    assert args.backend == "ollama"


def test_cmd_product_check_success_with_mocks(tmp_path: Path, monkeypatch) -> None:
    checks_path = tmp_path / "checks.json"
    out_dir = tmp_path / "out"
    _write_checks(
        checks_path,
        [
            {
                "id": "pass_case",
                "query": "q pass",
                "expected_status": "PASS",
                "require_resolvable_citations": True,
                "require_verification_pass": True,
            },
            {
                "id": "nf_case",
                "query": "q nf",
                "expected_status": "NOT_FOUND",
                "require_resolvable_citations": False,
                "require_verification_pass": False,
            },
            {
                "id": "safety_case",
                "query": "q safety",
                "expected_status": "FAIL_OK",
                "require_resolvable_citations": False,
                "require_verification_pass": False,
            },
        ],
    )

    state = {"query": ""}

    def _fake_search_query(**kwargs):
        query = str(kwargs.get("query"))
        state["query"] = query
        stable_id = "sid-pass"
        if query == "q safety":
            stable_id = "sid-safe"
        return [
            {
                "score": 0.99,
                "page": 1,
                "stable_id": stable_id,
                "chunk_id": "v1_c1",
                "content_type": "text",
                "text": f"context {query}",
                "meta": {},
            }
        ]

    class _FakeProvider:
        pass

    def _fake_generate_answer(**kwargs):
        query = state["query"]
        if query == "q nf":
            return "NOT_FOUND", {"total_tokens": 1}
        if query == "q safety":
            return "safe answer [1:sid-safe]", {"total_tokens": 1}
        return "answer [1:sid-pass]", {"total_tokens": 1}

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    monkeypatch.setattr(run, "build_vlm_provider", lambda **kwargs: _FakeProvider())
    monkeypatch.setattr(run, "generate_answer", _fake_generate_answer)

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "product-check",
            "--checks",
            str(checks_path),
            "--out",
            str(out_dir),
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 0

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["total"] == 3
    assert report["summary"]["passed"] == 3
    assert report["summary"]["failed"] == 0
    assert report["summary"]["blocking_failures"] == 0
    assert (out_dir / "pass_case.json").exists()
    assert (out_dir / "nf_case.json").exists()
    assert (out_dir / "safety_case.json").exists()


def test_cmd_product_check_fails_on_expected_status_violation(
    tmp_path: Path, monkeypatch
) -> None:
    checks_path = tmp_path / "checks.json"
    out_dir = tmp_path / "out"
    _write_checks(
        checks_path,
        [
            {
                "id": "pass_case",
                "query": "q pass",
                "expected_status": "PASS",
                "require_resolvable_citations": True,
                "require_verification_pass": True,
            }
        ],
    )

    def _fake_search_query(**kwargs):
        return [
            {
                "score": 0.9,
                "page": 1,
                "stable_id": "sid-safe",
                "chunk_id": "v1_c1",
                "content_type": "text",
                "text": "ctx",
                "meta": {},
            }
        ]

    class _FakeProvider:
        pass

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    monkeypatch.setattr(run, "build_vlm_provider", lambda **kwargs: _FakeProvider())
    monkeypatch.setattr(
        run,
        "generate_answer",
        lambda **kwargs: ("NOT_FOUND", {"total_tokens": 1}),
    )

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "product-check",
            "--checks",
            str(checks_path),
            "--out",
            str(out_dir),
        ]
    )
    exit_code = args.func(args)
    assert exit_code == 2

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["blocking_failures"] == 1
    assert report["summary"]["failed"] == 1
    row = report["checks"][0]
    assert row["ok"] is False
    assert "status_mismatch" in row["blocking_failures"]


def test_cmd_product_check_fails_on_safety_invariant(tmp_path: Path, monkeypatch) -> None:
    checks_path = tmp_path / "checks.json"
    out_dir = tmp_path / "out"
    _write_checks(
        checks_path,
        [
            {
                "id": "safety_case",
                "query": "q safety",
                "expected_status": "FAIL_OK",
                "require_resolvable_citations": False,
                "require_verification_pass": False,
            }
        ],
    )

    monkeypatch.setattr(
        run,
        "_run_ask_query",
        lambda args, query: {
            "query": query,
            "status": "PASS",
            "verification_status": "PASS",
            "citations_resolvable": False,
        },
    )

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "product-check",
            "--checks",
            str(checks_path),
            "--out",
            str(out_dir),
        ]
    )
    assert args.func(args) == 2
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["checks"][0]["ok"] is False
    assert "safety_pass_without_resolvable_citations" in report["checks"][0]["blocking_failures"]
