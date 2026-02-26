from __future__ import annotations

import json

import run


def test_ask_args_defaults() -> None:
    parser = run.build_parser()
    args = parser.parse_args(["ask", "--query", "q"])
    assert args.qdrant_url == "http://127.0.0.1:6333"
    assert args.collection == "bookmind_bench"
    assert args.top_k == 20
    assert args.content_types == "text,figure_caption"
    assert args.backend == "ollama"
    assert args.force_citations is False
    assert args.citation_min_count == 1
    assert args.citation_repair_retry == "auto"
    assert args.enforce_verified is False
    assert args.out is None


def test_cmd_ask_pass_with_citations(tmp_path, monkeypatch) -> None:
    out_path = tmp_path / "ask_result.json"

    def _fake_search_query(**kwargs):
        return [
            {
                "score": 0.99,
                "page": 1,
                "stable_id": "s1",
                "chunk_id": "v1_c1",
                "content_type": "text",
                "text": "pump pressure",
                "meta": {},
            }
        ]

    class _FakeProvider:
        pass

    monkeypatch.setattr(run, "search_query", _fake_search_query)
    monkeypatch.setattr(run, "build_vlm_provider", lambda **kwargs: _FakeProvider())
    monkeypatch.setattr(run, "generate_answer", lambda **kwargs: ("ok [1:s1]", {"total_tokens": 1}))

    parser = run.build_parser()
    args = parser.parse_args(["ask", "--query", "find x", "--out", str(out_path)])
    exit_code = args.func(args)

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["query"] == "find x"
    assert payload["status"] == "PASS"
    assert payload["verification_status"] == "PASS"
    assert payload["citations_total"] == 1
    assert payload["citations_resolved"] == 1
    assert payload["citations_resolvable"] is True
    assert payload["cited_pages"] == [1]
    assert payload["cited_ids"] == ["s1"]
    assert payload["first_cited_rank"] == 1
    assert payload["retrieved_rows"][0]["rank"] == 1


def test_cmd_ask_not_found_path(tmp_path, monkeypatch) -> None:
    out_path = tmp_path / "ask_result.json"

    def _fake_search_query(**kwargs):
        return [
            {
                "score": 0.99,
                "page": 1,
                "stable_id": "s1",
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
    monkeypatch.setattr(run, "generate_answer", lambda **kwargs: ("NOT_FOUND", {"total_tokens": 1}))

    parser = run.build_parser()
    args = parser.parse_args(["ask", "--query", "find x", "--out", str(out_path)])
    exit_code = args.func(args)

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "NOT_FOUND"
    assert payload["final_answer"] == "NOT_FOUND"
    assert payload["verification_status"] == "PASS"
    assert payload["verification_reason"] == "NOT_FOUND"


def test_cmd_ask_fail_unresolvable_citations(tmp_path, monkeypatch) -> None:
    out_path = tmp_path / "ask_result.json"

    def _fake_search_query(**kwargs):
        return [
            {
                "score": 0.99,
                "page": 1,
                "stable_id": "s1",
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
    monkeypatch.setattr(run, "generate_answer", lambda **kwargs: ("answer [1:s2]", {"total_tokens": 1}))

    parser = run.build_parser()
    args = parser.parse_args(["ask", "--query", "find x", "--out", str(out_path)])
    exit_code = args.func(args)

    assert exit_code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["verification_status"] == "FAIL_UNRESOLVABLE_CITATIONS"
    assert payload["citations_total"] == 1
    assert payload["citations_resolved"] == 0
    assert payload["citations_resolvable"] is False


def test_cmd_ask_enforce_verified_exits_nonzero(tmp_path, monkeypatch) -> None:
    out_path = tmp_path / "ask_result.json"

    def _fake_search_query(**kwargs):
        return [
            {
                "score": 0.99,
                "page": 1,
                "stable_id": "s1",
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
    monkeypatch.setattr(run, "generate_answer", lambda **kwargs: ("answer [1:s2]", {"total_tokens": 1}))

    parser = run.build_parser()
    args = parser.parse_args(
        [
            "ask",
            "--query",
            "find x",
            "--out",
            str(out_path),
            "--enforce_verified",
            "true",
        ]
    )
    exit_code = args.func(args)

    assert exit_code == 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["verification_status"] == "FAIL_UNRESOLVABLE_CITATIONS"
