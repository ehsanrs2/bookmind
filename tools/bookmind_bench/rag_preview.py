"""Bench-only RAG preview helpers built on Qdrant search results."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from engines.vlm_providers import OllamaProvider, OpenAICompatProvider

DEFAULT_SYSTEM_PROMPT = (
    "You are a technical assistant. Provide concise, factual answers grounded in the "
    "provided document context."
)


def _truncate_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


def _build_record_text(row: Dict[str, Any], max_chars: int = 500) -> str:
    content_type = str(row.get("content_type") or "")
    text = str(row.get("text") or "")
    if content_type == "figure_caption":
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        figure_context = meta.get("figure_context")
        if figure_context:
            merged = f"Figure context: {figure_context}\nCaption: {text}"
            return _truncate_text(merged, max_chars)
    return _truncate_text(text, max_chars)


def build_context(results: Sequence[Dict[str, Any]], max_chars: int = 6000) -> Dict[str, Any]:
    snippets: List[str] = []
    citations: List[Dict[str, Any]] = []
    total_len = 0
    delimiter = "\n\n---\n\n"

    for rank, row in enumerate(results, start=1):
        stable_id = str(row.get("stable_id") or row.get("id") or "")
        page = _to_int(row.get("page"))
        content_type = str(row.get("content_type") or "unknown")
        bbox = row.get("bbox") if isinstance(row.get("bbox"), list) else None
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        figure_ref = meta.get("figure_ref") if isinstance(meta.get("figure_ref"), dict) else None

        body = _build_record_text(row, max_chars=500)
        header = (
            f"[{rank}] (page:{page if page is not None else '?'}) "
            f"(content_type:{content_type}) stable_id:{stable_id}"
        )
        snippet = f"{header}\n{body}"
        piece_len = len(snippet) + (len(delimiter) if snippets else 0)
        if snippets and total_len + piece_len > max_chars:
            break
        if not snippets and len(snippet) > max_chars:
            break

        snippets.append(snippet)
        total_len += piece_len
        citations.append(
            {
                "rank": rank,
                "stable_id": stable_id,
                "page": page,
                "content_type": content_type,
                "bbox": bbox,
                "figure_ref": figure_ref,
                "snippet": body,
            }
        )

    context_text = delimiter.join(snippets)
    if len(context_text) > max_chars:
        context_text = context_text[:max_chars]

    return {
        "context_text": context_text,
        "citations": citations,
    }


def build_rag_messages(query: str, context_text: str) -> List[Dict[str, str]]:
    user_content = (
        "User query:\n"
        f"{query.strip()}\n\n"
        "Retrieved context:\n"
        "```\n"
        f"{context_text.strip()}\n"
        "```\n\n"
        "Instructions:\n"
        "- Answer using only the provided context.\n"
        "- If the context is insufficient, explicitly say the context is insufficient.\n"
        "- Include citations in the form [page:stable_id]."
    )
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_answer(
    provider: Any,
    messages: Sequence[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    if hasattr(provider, "chat") and callable(provider.chat):
        return provider.chat(messages=messages, max_tokens=max_tokens, temperature=temperature)

    if isinstance(provider, OpenAICompatProvider):
        payload = {
            "model": provider.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = provider.endpoint + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=provider.timeout_s) as response:
            parsed = json.loads(response.read().decode("utf-8"))

        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI-compatible response missing choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise ValueError("OpenAI-compatible response missing message")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("OpenAI-compatible response missing content")
        usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None
        return content.strip(), usage

    if isinstance(provider, OllamaProvider):
        payload = {
            "model": provider.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        response = requests.post(
            provider.ollama_url + "/api/chat",
            json=payload,
            timeout=provider.timeout_s,
        )
        response.raise_for_status()
        parsed = response.json()
        content: Optional[str] = None
        message = parsed.get("message")
        if isinstance(message, dict):
            message_content = message.get("content")
            if isinstance(message_content, str):
                content = message_content
        if content is None:
            response_text = parsed.get("response")
            if isinstance(response_text, str):
                content = response_text
        if content is None:
            keys = sorted(str(key) for key in parsed.keys()) if isinstance(parsed, dict) else []
            raise ValueError(
                "Ollama response missing assistant content "
                "(expected message.content or response). "
                f"response_keys={keys}"
            )
        usage = {
            "prompt_eval_count": parsed.get("prompt_eval_count"),
            "eval_count": parsed.get("eval_count"),
            "prompt_eval_duration": parsed.get("prompt_eval_duration"),
            "eval_duration": parsed.get("eval_duration"),
            "total_duration": parsed.get("total_duration"),
        }
        return content.strip(), usage

    raise TypeError(f"Unsupported provider type for RAG preview: {type(provider)!r}")


def format_preview_output(
    answer_text: str,
    citations: Sequence[Dict[str, Any]],
    show_snippets: bool = False,
) -> str:
    lines: List[str] = []
    lines.append("Answer")
    lines.append("======")
    lines.append(str(answer_text or "").strip() or "(empty answer)")
    lines.append("")
    lines.append("Citations")
    lines.append("=========")

    if not citations:
        lines.append("(none)")
    else:
        for citation in citations:
            rank = citation.get("rank")
            page = citation.get("page")
            content_type = citation.get("content_type")
            stable_id = citation.get("stable_id")
            lines.append(
                f"- rank={rank} page={page} type={content_type} stable_id={stable_id}"
            )
            figure_ref = citation.get("figure_ref")
            if isinstance(figure_ref, dict):
                crop_image = figure_ref.get("crop_image")
                page_image = figure_ref.get("page_image")
                if crop_image or page_image:
                    lines.append(
                        f"  figure_ref crop_image={crop_image} page_image={page_image}"
                    )

    if show_snippets:
        lines.append("")
        lines.append("Retrieved snippets")
        lines.append("==================")
        if not citations:
            lines.append("(none)")
        else:
            for citation in citations:
                lines.append(
                    f"[{citation.get('rank')}] {citation.get('snippet') or ''}".strip()
                )

    return "\n".join(lines).rstrip() + "\n"
