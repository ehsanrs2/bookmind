from rag_preview import (
    apply_force_citation_repair_fallback,
    build_citation_repair_messages,
    build_force_citation_allowlist,
    build_rag_messages,
)


def test_build_rag_messages_contains_required_instruction() -> None:
    query = "What are calibration conditions?"
    context = "[1] (page:2) (content_type:text) stable_id:x\\nA context chunk"

    messages = build_rag_messages(query, context)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "technical assistant" in messages[0]["content"].lower()

    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert query in user_content
    assert context in user_content
    assert "only the provided context" in user_content
    assert "context is insufficient" in user_content
    assert "[page:stable_id]" in user_content


def test_build_rag_messages_force_citations_injects_strict_policy() -> None:
    allowlist = build_force_citation_allowlist(
        [
            {"stable_id": "s1", "chunk_id": "v1_chunk_1"},
            {"stable_id": "s2"},
        ]
    )
    messages = build_rag_messages(
        "What is X?",
        "[1] (page:1) (content_type:text) stable_id:s1\nX is present.",
        force_citations=True,
        citation_min_count=2,
        allowed_citation_ids=allowlist,
    )

    user_content = messages[1]["content"]
    assert "MUST include citation markers" in user_content
    assert "Include at least 2 citation(s)." in user_content
    assert "reply exactly: NOT_FOUND" in user_content
    assert "Allowed citation IDs (use ONLY these): v1_chunk_1, s2" in user_content
    assert "IDs are case-sensitive and must be copied exactly." in user_content
    assert "Every citation marker MUST use an ID from the allowlist above." in user_content


def test_build_rag_messages_allowlist_only_in_force_mode() -> None:
    allowlist = ["v1_chunk_1", "s2"]
    messages = build_rag_messages(
        "What is X?",
        "[1] (page:1) (content_type:text) stable_id:s1\nX is present.",
        force_citations=False,
        citation_min_count=1,
        allowed_citation_ids=allowlist,
    )
    assert "Allowed citation IDs (use ONLY these):" not in messages[1]["content"]


def test_build_force_citation_allowlist_prefers_chunk_id() -> None:
    allowlist = build_force_citation_allowlist(
        [
            {"rank": 1, "stable_id": "S1", "chunk_id": "v1_A"},
            {"rank": 2, "stable_id": "S2", "chunk_id": ""},
            {"rank": 3, "stable_id": "S3"},
        ]
    )
    assert allowlist == ["v1_A", "S2", "S3"]


def test_build_citation_repair_messages_includes_allowlist_constraint() -> None:
    messages = build_citation_repair_messages(
        query="What is X?",
        context_text="ctx",
        original_answer="X",
        citation_min_count=1,
        allowed_citation_ids=["v1_A", "S2"],
    )
    user_content = messages[1]["content"]
    assert "Every factual sentence ends with a citation marker." in user_content
    assert "Citation markers must appear at the END of sentences." in user_content
    assert "If you cannot cite a sentence, remove that sentence." in user_content
    assert "Use ONLY allowed citation IDs listed in the context. Do not invent." in user_content
    assert "Allowed citation IDs (use ONLY these): v1_A, S2" in user_content


def test_apply_force_citation_repair_fallback_appends_sources_line() -> None:
    repaired = apply_force_citation_repair_fallback(
        "The valve opens during startup.",
        citation_min_count=1,
        allowed_citation_ids=["v1_chunk_s1"],
    )
    assert repaired.endswith("Sources: [v1_chunk_s1]")


def test_apply_force_citation_repair_fallback_preserves_not_found() -> None:
    repaired = apply_force_citation_repair_fallback(
        "NOT_FOUND",
        citation_min_count=1,
        allowed_citation_ids=["v1_chunk_s1"],
    )
    assert repaired == "NOT_FOUND"
