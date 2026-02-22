"""Additive answer verifier for citation validity and grounding checks."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

PASS = "PASS"
FAIL_NO_CITATIONS = "FAIL_NO_CITATIONS"
FAIL_UNRESOLVABLE_CITATIONS = "FAIL_UNRESOLVABLE_CITATIONS"
FAIL_UNSUPPORTED_CLAIMS = "FAIL_UNSUPPORTED_CLAIMS"

_SAFE_ANSWER = "I couldn't find this in the provided text."
_WORD_RE = re.compile(r"[a-zA-Z0-9]{3,}")
_CITATION_RE = re.compile(r"\[[^\]]+\]")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "into",
    "onto",
    "about",
    "there",
    "their",
    "have",
    "has",
    "had",
    "were",
    "was",
    "are",
    "is",
    "be",
    "been",
    "being",
    "what",
    "when",
    "where",
    "which",
    "while",
    "would",
    "could",
    "should",
    "can",
    "may",
    "might",
    "will",
    "your",
    "you",
    "they",
    "them",
    "its",
    "our",
    "his",
    "her",
    "not",
    "only",
    "also",
    "than",
    "then",
    "any",
    "all",
    "some",
    "more",
    "most",
    "much",
    "many",
    "few",
    "such",
    "very",
    "just",
    "over",
    "under",
    "between",
    "through",
    "using",
    "used",
    "use",
    "based",
    "context",
    "provided",
    "document",
    "answer",
}


def _retrieved_text_blob(retrieved_rows: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in retrieved_rows:
        if not isinstance(row, dict):
            continue
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.lower())
            continue
        snippet = row.get("snippet")
        if isinstance(snippet, str) and snippet.strip():
            parts.append(snippet.lower())
    return "\n".join(parts)


def _extract_key_words(answer_text: str) -> List[str]:
    cleaned = _CITATION_RE.sub(" ", str(answer_text or "").lower())
    words = [w for w in _WORD_RE.findall(cleaned) if w not in _STOPWORDS]
    deduped: List[str] = []
    seen = set()
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        deduped.append(word)
        if len(deduped) >= 10:
            break
    return deduped


def verify_answer(
    *,
    answer_text: str,
    retrieved_rows: Sequence[Dict[str, Any]],
    citation_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Return additive verification result for one generated answer."""
    citations_total = int(citation_report.get("citations_total") or 0)
    if citations_total == 0:
        return {
            "verification_status": FAIL_NO_CITATIONS,
            "short_reason": "Model answer has no citation markers.",
            "suggested_safe_answer": _SAFE_ANSWER,
        }

    if not bool(citation_report.get("citations_resolvable")):
        return {
            "verification_status": FAIL_UNRESOLVABLE_CITATIONS,
            "short_reason": "At least one citation does not resolve to retrieved context.",
            "suggested_safe_answer": _SAFE_ANSWER,
        }

    key_words = _extract_key_words(answer_text)
    if len(key_words) < 3:
        # Conservative path: allow terse answers once citations are valid.
        return {
            "verification_status": PASS,
            "short_reason": "Citations resolve; grounding check skipped for terse answer.",
        }

    context_blob = _retrieved_text_blob(retrieved_rows)
    overlap = [word for word in key_words if word in context_blob]
    min_overlap = 2 if len(key_words) >= 6 else 1
    if len(overlap) < min_overlap:
        return {
            "verification_status": FAIL_UNSUPPORTED_CLAIMS,
            "short_reason": (
                f"Low grounding overlap: matched {len(overlap)} of {len(key_words)} key terms "
                f"(required >= {min_overlap})."
            ),
            "suggested_safe_answer": _SAFE_ANSWER,
        }

    return {
        "verification_status": PASS,
        "short_reason": (
            f"Citations resolve and grounding overlap is sufficient ({len(overlap)}/{len(key_words)})."
        ),
    }
