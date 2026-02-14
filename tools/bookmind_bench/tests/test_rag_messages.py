from rag_preview import build_rag_messages


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
