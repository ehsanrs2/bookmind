"""Citation parsing and resolution helpers for RAG outputs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

_CITATION_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


def _append_page_or_id_citation(
    parsed: List[Dict[str, Any]],
    *,
    raw: str,
    page: int | None,
    stable_id: str | None = None,
    chunk_id: str | None = None,
    source_id: str | None = None,
) -> None:
    stable = str(stable_id or "").strip()
    chunk = str(chunk_id or "").strip()
    source = str(source_id or "").strip()
    if not stable and not chunk:
        return
    if chunk:
        if page is not None:
            kind = "page_chunk_id"
        elif source:
            kind = "source_chunk_id"
        else:
            kind = "chunk_id"
        parsed.append(
            {
                "kind": kind,
                "page": page,
                "stable_id": None,
                "chunk_id": chunk,
                "source_id": source or None,
                "raw": raw,
            }
        )
        return
    kind = "page_stable_id" if page is not None else "stable_id"
    parsed.append(
        {
            "kind": kind,
            "page": page,
            "stable_id": stable,
            "chunk_id": None,
            "source_id": source or None,
            "raw": raw,
        }
    )


def _parse_key_value_tokens(body: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in str(body or "").split(","):
        item = token.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            continue
        out[key] = value
    return out


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


def parse_citations(text: str) -> List[Dict[str, Any]]:
    """Parse citation markers from answer text.

    Supported formats:
    - [page:stable_id]
    - [page:chunk_id] where chunk_id starts with v1_
    - [source_id:chunk_id] where chunk_id starts with v1_
    """
    parsed: List[Dict[str, Any]] = []
    for match in _CITATION_BRACKET_RE.finditer(str(text or "")):
        raw = match.group(0)
        body = match.group(1).strip()
        if not body or ":" not in body:
            continue

        key_values = _parse_key_value_tokens(body)
        if key_values:
            page_value = _to_int(key_values.get("page"))
            stable_value = str(key_values.get("stable_id") or "").strip()
            chunk_value = str(key_values.get("chunk_id") or "").strip()
            if chunk_value and chunk_value.startswith("v1_"):
                _append_page_or_id_citation(
                    parsed,
                    raw=raw,
                    page=page_value,
                    chunk_id=chunk_value,
                    source_id=key_values.get("source_id"),
                )
                continue
            if stable_value:
                _append_page_or_id_citation(
                    parsed,
                    raw=raw,
                    page=page_value,
                    stable_id=stable_value,
                    source_id=key_values.get("source_id"),
                )
                continue

        left, right = body.split(":", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            continue

        page = _to_int(left)
        if page is not None:
            if right.startswith("v1_"):
                _append_page_or_id_citation(parsed, raw=raw, page=page, chunk_id=right)
            else:
                _append_page_or_id_citation(parsed, raw=raw, page=page, stable_id=right)
            continue

        if left.lower() == "page":
            if right.startswith("v1_"):
                _append_page_or_id_citation(parsed, raw=raw, page=None, chunk_id=right)
            elif _to_int(right) is None:
                _append_page_or_id_citation(parsed, raw=raw, page=None, stable_id=right)
            continue

        if left.lower() == "stable_id":
            _append_page_or_id_citation(parsed, raw=raw, page=None, stable_id=right)
            continue

        if left.lower() == "chunk_id" and right.startswith("v1_"):
            _append_page_or_id_citation(parsed, raw=raw, page=None, chunk_id=right)
            continue

        if right.startswith("v1_"):
            _append_page_or_id_citation(parsed, raw=raw, page=None, chunk_id=right, source_id=left)

    return parsed


def resolve_citations(
    citations: Sequence[Dict[str, Any]],
    retrieved_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve parsed citations against retrieved rows used in prompt context."""
    unresolved: List[Dict[str, Any]] = []
    resolved_ids: List[str] = []
    resolved_pages: set[int] = set()
    cited_ranks: List[int] = []

    for citation in citations:
        if not isinstance(citation, dict):
            unresolved.append(
                {
                    "kind": None,
                    "page": None,
                    "stable_id": None,
                    "chunk_id": None,
                    "source_id": None,
                    "raw": str(citation),
                }
            )
            continue

        citation_page = _to_int(citation.get("page"))
        citation_chunk_id = str(citation.get("chunk_id") or "")
        citation_stable_id = str(citation.get("stable_id") or "")

        matches: List[Dict[str, Any]] = []
        for rank, row in enumerate(retrieved_rows, start=1):
            if not isinstance(row, dict):
                continue
            row_chunk_id = str(row.get("chunk_id") or "")
            row_stable_id = str(row.get("stable_id") or row.get("id") or "")
            row_page = _to_int(row.get("page"))

            is_match = False
            if citation_chunk_id:
                is_match = row_chunk_id == citation_chunk_id
            elif citation_stable_id:
                is_match = row_stable_id == citation_stable_id

            if not is_match:
                continue
            if citation_page is not None and row_page is not None and row_page != citation_page:
                continue

            candidate = dict(row)
            candidate["_rank"] = rank
            matches.append(candidate)

        if not matches:
            unresolved.append(citation)
            continue

        best = min(matches, key=lambda row: int(row.get("_rank") or 10**9))
        matched_rank = _to_int(best.get("_rank"))
        if matched_rank is not None:
            cited_ranks.append(matched_rank)

        resolved_page = _to_int(best.get("page"))
        if resolved_page is not None:
            resolved_pages.add(resolved_page)

        matched_id = ""
        if citation_chunk_id:
            matched_id = str(best.get("chunk_id") or citation_chunk_id)
        elif citation_stable_id:
            matched_id = str(best.get("stable_id") or best.get("id") or citation_stable_id)
        if matched_id and matched_id not in resolved_ids:
            resolved_ids.append(matched_id)

    citations_total = len([citation for citation in citations if isinstance(citation, dict)])
    citations_resolved = citations_total - len(unresolved)
    citations_resolvable = citations_total == citations_resolved
    first_cited_rank = min(cited_ranks) if citations_resolvable and cited_ranks else None

    return {
        "citations_total": citations_total,
        "citations_resolved": citations_resolved,
        "citations_unresolved": unresolved,
        "citations_resolvable": citations_resolvable,
        "resolved_ids": resolved_ids,
        "resolved_pages": resolved_pages,
        "first_cited_rank": first_cited_rank,
    }
