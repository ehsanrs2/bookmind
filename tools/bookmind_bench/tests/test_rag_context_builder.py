from rag_preview import build_context


def test_build_context_enforces_bounds_and_citations_fields() -> None:
    results = [
        {
            "stable_id": "text_001",
            "page": 3,
            "bbox": [1, 2, 3, 4],
            "content_type": "text",
            "text": "A" * 1200,
            "meta": {},
        },
        {
            "stable_id": "fig_002",
            "page": 5,
            "bbox": [10, 20, 30, 40],
            "content_type": "figure_caption",
            "text": "caption body",
            "meta": {
                "figure_context": "nearby figure context",
                "figure_ref": {
                    "crop_image": "bundle/images/crop_2.png",
                    "page_image": "bundle/images/page_5.png",
                },
            },
        },
    ]

    payload = build_context(results, max_chars=1200)
    context_text = payload["context_text"]
    citations = payload["citations"]

    assert "stable_id:text_001" in context_text
    assert "(page:3)" in context_text
    assert "stable_id:fig_002" in context_text
    assert "Figure context: nearby figure context" in context_text
    assert len(context_text) <= 1200

    assert len(citations) == 2
    first = citations[0]
    assert first["rank"] == 1
    assert first["stable_id"] == "text_001"
    assert first["page"] == 3
    assert first["content_type"] == "text"
    assert first["bbox"] == [1, 2, 3, 4]

    second = citations[1]
    assert second["rank"] == 2
    assert second["stable_id"] == "fig_002"
    assert second["figure_ref"] is not None
    assert second["figure_ref"]["crop_image"] == "bundle/images/crop_2.png"


def test_build_context_respects_max_chars_by_snippet_boundary() -> None:
    results = [
        {
            "stable_id": "a",
            "page": 1,
            "bbox": [0, 0, 1, 1],
            "content_type": "text",
            "text": "first snippet content",
            "meta": {},
        },
        {
            "stable_id": "b",
            "page": 2,
            "bbox": [0, 0, 1, 1],
            "content_type": "text",
            "text": "second snippet content",
            "meta": {},
        },
    ]

    payload = build_context(results, max_chars=90)
    assert len(payload["context_text"]) <= 90
    assert len(payload["citations"]) <= 1
