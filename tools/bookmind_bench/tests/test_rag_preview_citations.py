from __future__ import annotations

from rag_preview import build_model_citation_report, format_preview_output


def test_format_preview_output_includes_model_citation_section() -> None:
    retrieved = [
        {
            "rank": 1,
            "stable_id": "s1",
            "chunk_id": "v1_c1",
            "page": 1,
            "content_type": "text",
            "snippet": "ctx",
        }
    ]
    report = build_model_citation_report("Answer [1:s1]", retrieved)

    out = format_preview_output(
        answer_text="Answer [1:s1]",
        citations=retrieved,
        model_citations=report,
        show_snippets=False,
    )

    assert "Model citations" in out
    assert "citations_total=1" in out
    assert "citations_resolved=1" in out
    assert "citations_resolvable=True" in out


def test_format_preview_output_shows_unresolved_preview() -> None:
    report = build_model_citation_report("Answer [3:s9]", [{"stable_id": "s1", "chunk_id": "v1_c1", "page": 1}])
    out = format_preview_output(answer_text="Answer", citations=[], model_citations=report)

    assert "citations_resolvable=False" in out
    assert "- unresolved:" in out
    assert "[3:s9]" in out
