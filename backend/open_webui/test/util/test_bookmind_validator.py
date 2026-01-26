from pathlib import Path

from open_webui.utils.bookmind_validator import load_json, validate_scope

EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "test" / "test_files" / "bookmind"


def test_scope_examples_validate():
    examples = sorted(EXAMPLES_DIR.glob("*.json"))
    assert examples, "No Bookmind scope examples found"
    for example_path in examples:
        scope = load_json(example_path)
        ok, errors = validate_scope(scope)
        assert ok, f"{example_path.name} failed: {errors}"


def test_book_qa_requires_citations():
    scope = load_json(EXAMPLES_DIR / "book_qa_whole_book.json")
    scope["citations_required"] = False
    ok, errors = validate_scope(scope)
    assert not ok
    assert errors


def test_free_chat_disables_retrieval():
    scope = load_json(EXAMPLES_DIR / "free_chat.json")
    scope["retrieval"]["enabled"] = True
    ok, errors = validate_scope(scope)
    assert not ok
    assert errors
