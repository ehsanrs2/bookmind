from verifier import (
    FAIL_NO_CITATIONS,
    FAIL_UNRESOLVABLE_CITATIONS,
    FAIL_UNSUPPORTED_CLAIMS,
    PASS,
    verify_answer,
)


def test_verify_answer_pass_with_resolved_citations_and_overlap() -> None:
    report = verify_answer(
        answer_text="The Mach indicator connects pitot and static pressure lines [4:s1].",
        retrieved_rows=[{"text": "Mach indicator uses pitot and static pressure lines."}],
        citation_report={"citations_total": 1, "citations_resolvable": True},
    )
    assert report["verification_status"] == PASS


def test_verify_answer_fails_without_citations() -> None:
    report = verify_answer(
        answer_text="Some answer without markers",
        retrieved_rows=[{"text": "context text"}],
        citation_report={"citations_total": 0, "citations_resolvable": True},
    )
    assert report["verification_status"] == FAIL_NO_CITATIONS
    assert "suggested_safe_answer" in report


def test_verify_answer_not_found_passes_without_citations() -> None:
    report = verify_answer(
        answer_text=" NOT_FOUND ",
        retrieved_rows=[{"text": "context text"}],
        citation_report={"citations_total": 0, "citations_resolvable": False},
    )
    assert report["verification_status"] == PASS
    assert report["short_reason"] == "NOT_FOUND"
    assert report["suggested_safe_answer"] is None


def test_verify_answer_fails_unresolvable_citations() -> None:
    report = verify_answer(
        answer_text="Mapped signal [3:s9].",
        retrieved_rows=[{"text": "signal path"}],
        citation_report={"citations_total": 1, "citations_resolvable": False},
    )
    assert report["verification_status"] == FAIL_UNRESOLVABLE_CITATIONS


def test_verify_answer_fails_on_low_grounding_overlap() -> None:
    report = verify_answer(
        answer_text="Quantum lattice entropy manifold calibration tensor [1:s1].",
        retrieved_rows=[{"text": "Pitot static pressure system and mach indicator."}],
        citation_report={"citations_total": 1, "citations_resolvable": True},
    )
    assert report["verification_status"] == FAIL_UNSUPPORTED_CLAIMS
    assert "Low grounding overlap" in str(report["short_reason"])
