from __future__ import annotations

from citations import parse_citations, resolve_citations


def test_parse_citations_supports_stable_and_chunk_formats() -> None:
    text = "Answer [12:sid_12] and [9:v1_chunk_9] and [src_a:v1_chunk_2]"
    citations = parse_citations(text)

    assert len(citations) == 3
    assert citations[0]["kind"] == "page_stable_id"
    assert citations[0]["page"] == 12
    assert citations[0]["stable_id"] == "sid_12"

    assert citations[1]["kind"] == "page_chunk_id"
    assert citations[1]["page"] == 9
    assert citations[1]["chunk_id"] == "v1_chunk_9"

    assert citations[2]["kind"] == "source_chunk_id"
    assert citations[2]["source_id"] == "src_a"
    assert citations[2]["chunk_id"] == "v1_chunk_2"


def test_parse_citations_is_robust_to_noise_and_repeats() -> None:
    text = "noise [] [abc] [1:] [:x] [1:s1] [1:s1] [broken [2:v1_c2] tail]"
    citations = parse_citations(text)

    assert len(citations) == 3
    assert citations[0]["raw"] == "[1:s1]"
    assert citations[1]["raw"] == "[1:s1]"
    assert citations[2]["raw"] == "[2:v1_c2]"


def test_parse_citations_supports_page_keyword_and_key_value_formats() -> None:
    text = (
        "A [page:32522d73d6d3a42c] "
        "B [page:5, stable_id:0b3dcdd038dd0db6] "
        "C [page:5, chunk_id:v1_abcdef123456] "
        "D [stable_id:feedface] "
        "E [chunk_id:v1_deadbeef]"
    )
    citations = parse_citations(text)

    assert len(citations) == 5
    assert citations[0]["kind"] == "stable_id"
    assert citations[0]["page"] is None
    assert citations[0]["stable_id"] == "32522d73d6d3a42c"
    assert citations[1]["kind"] == "page_stable_id"
    assert citations[1]["page"] == 5
    assert citations[1]["stable_id"] == "0b3dcdd038dd0db6"
    assert citations[2]["kind"] == "page_chunk_id"
    assert citations[2]["page"] == 5
    assert citations[2]["chunk_id"] == "v1_abcdef123456"
    assert citations[3]["kind"] == "stable_id"
    assert citations[3]["stable_id"] == "feedface"
    assert citations[4]["kind"] == "chunk_id"
    assert citations[4]["chunk_id"] == "v1_deadbeef"


def test_resolve_citations_matches_chunk_id_and_stable_id() -> None:
    retrieved = [
        {"stable_id": "s1", "chunk_id": "v1_c1", "page": 1},
        {"stable_id": "s2", "chunk_id": "v1_c2", "page": 2},
    ]
    citations = parse_citations("A [1:s1] B [2:v1_c2]")

    resolved = resolve_citations(citations, retrieved)

    assert resolved["citations_total"] == 2
    assert resolved["citations_resolved"] == 2
    assert resolved["citations_resolvable"] is True
    assert resolved["resolved_ids"] == ["s1", "v1_c2"]
    assert resolved["resolved_pages"] == {1, 2}
    assert resolved["first_cited_rank"] == 1


def test_resolve_citations_supports_page_keyword_and_missing_row_page() -> None:
    retrieved = [
        {"stable_id": "32522d73d6d3a42c", "chunk_id": "v1_a", "page": None},
        {"stable_id": "0b3dcdd038dd0db6", "chunk_id": "v1_b", "page": 5},
        {"stable_id": "other", "chunk_id": "v1_abcdef123456", "page": 5},
    ]
    citations = parse_citations(
        "A [page:32522d73d6d3a42c] B [page:5, stable_id:0b3dcdd038dd0db6] C [page:5, chunk_id:v1_abcdef123456]"
    )

    resolved = resolve_citations(citations, retrieved)

    assert resolved["citations_total"] == 3
    assert resolved["citations_resolved"] == 3
    assert resolved["citations_resolvable"] is True
    assert "32522d73d6d3a42c" in resolved["resolved_ids"]
    assert "0b3dcdd038dd0db6" in resolved["resolved_ids"]
    assert "v1_abcdef123456" in resolved["resolved_ids"]


def test_resolve_citations_reports_unresolved_items() -> None:
    retrieved = [{"stable_id": "s1", "chunk_id": "v1_c1", "page": 1}]
    citations = parse_citations("A [1:s1] B [2:s1] C [3:v1_missing]")

    resolved = resolve_citations(citations, retrieved)

    assert resolved["citations_total"] == 3
    assert resolved["citations_resolved"] == 1
    assert resolved["citations_resolvable"] is False
    assert len(resolved["citations_unresolved"]) == 2
    assert resolved["first_cited_rank"] is None
