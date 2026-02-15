"""Bench-only RAG preview helpers built on Qdrant search results."""

from __future__ import annotations

import copy
import json
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests

from engines.vlm_providers import (
    OllamaProvider,
    OpenAICompatProvider,
    extract_ollama_usage,
    parse_ollama_chat_payload,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a technical assistant. Provide concise, factual answers grounded in the "
    "provided document context."
)
OLLAMA_RETRY_SYSTEM_PROMPT = (
    "Answer only from retrieved context. If context is insufficient, say so plainly."
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


def _truncate_rag_user_context(content: str, max_context_chars: int) -> str:
    marker_start = "Retrieved context:\n```\n"
    marker_end = "\n```"
    text = str(content or "")
    start_idx = text.find(marker_start)
    if start_idx < 0:
        return _truncate_text(text, max_context_chars)
    context_start = start_idx + len(marker_start)
    end_idx = text.find(marker_end, context_start)
    if end_idx < 0:
        return _truncate_text(text, max_context_chars)
    context_body = text[context_start:end_idx]
    if len(context_body) <= max_context_chars:
        return text
    truncated = _truncate_text(context_body, max_context_chars)
    return text[:context_start] + truncated + text[end_idx:]


def _prepare_ollama_retry_messages(
    messages: Sequence[Dict[str, str]],
    *,
    context_limit: Optional[int] = None,
    simplify_system_prompt: bool = False,
) -> List[Dict[str, str]]:
    prepared: List[Dict[str, str]] = [copy.deepcopy(dict(msg)) for msg in messages]
    if simplify_system_prompt:
        for msg in prepared:
            if str(msg.get("role") or "") == "system":
                msg["content"] = OLLAMA_RETRY_SYSTEM_PROMPT
                break
    if context_limit is not None and context_limit > 0:
        for msg in prepared:
            if str(msg.get("role") or "") == "user":
                user_content = msg.get("content")
                if isinstance(user_content, str):
                    msg["content"] = _truncate_rag_user_context(
                        user_content, max_context_chars=context_limit
                    )
                break
    return prepared


def _ollama_header_subset(headers: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    keys = (
        "content-type",
        "content-length",
        "date",
        "server",
        "x-request-id",
        "x-ollama-model",
        "x-ollama-version",
    )
    header_items = {}
    if hasattr(headers, "items"):
        try:
            header_items = {str(k).lower(): str(v) for k, v in headers.items()}
        except Exception:
            header_items = {}
    for key in keys:
        value = header_items.get(key)
        if value is not None:
            out[key] = value
    return out


def _decode_ollama_response(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        raw_text = response.text
        parsed = parse_ollama_chat_payload(raw_text)
        return parsed.get("raw")


def generate_answer(
    provider: Any,
    messages: Sequence[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.2,
    ollama_debug_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
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
        attempts = [
            {
                "name": "initial",
                "context_limit": None,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "simplify_system_prompt": False,
            },
            {
                "name": "retry_context_reduced",
                "context_limit": 2500,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "simplify_system_prompt": False,
            },
            {
                "name": "retry_safe_prompt",
                "context_limit": 2500,
                "max_tokens": min(int(max_tokens), 256),
                "temperature": 0.0,
                "simplify_system_prompt": True,
            },
        ]
        last_error: Optional[Exception] = None
        for attempt_index, attempt in enumerate(attempts):
            attempt_messages = _prepare_ollama_retry_messages(
                messages,
                context_limit=attempt["context_limit"],
                simplify_system_prompt=attempt["simplify_system_prompt"],
            )
            options = {
                "num_predict": attempt["max_tokens"],
                "temperature": attempt["temperature"],
                "num_ctx": 4096,
            }
            payload = {
                "model": provider.model,
                "messages": attempt_messages,
                "stream": False,
                "options": options,
            }
            attempt_debug: Dict[str, Any] = {
                "attempt_index": attempt_index,
                "attempt_name": attempt["name"],
                "request": payload,
            }
            started = time.perf_counter()
            response = None
            parsed: Any = None
            try:
                response = requests.post(
                    provider.ollama_url + "/api/chat",
                    json=payload,
                    timeout=provider.timeout_s,
                )
                attempt_debug["http_status"] = int(response.status_code)
                attempt_debug["response_headers"] = _ollama_header_subset(response.headers)
                response.raise_for_status()
                parsed = _decode_ollama_response(response)
                parsed_payload = parse_ollama_chat_payload(parsed)
                content = parsed_payload.get("answer_text")
                cleaned = str(content or "").strip()
                usage = extract_ollama_usage(parsed)
                attempt_debug["response"] = {
                    "raw": parsed_payload.get("raw"),
                    "parsed_keys": parsed_payload.get("parsed_keys"),
                    "content_source": parsed_payload.get("source"),
                    "done_reason": parsed_payload.get("done_reason"),
                    "provider_error": parsed_payload.get("error"),
                    "answer_len_chars": len(cleaned),
                    "usage": usage,
                }
                if cleaned:
                    usage_with_attempt = dict(usage)
                    usage_with_attempt["attempt_index"] = attempt_index
                    usage_with_attempt["attempt_name"] = attempt["name"]
                    return cleaned, usage_with_attempt
                last_error = RuntimeError(
                    "Ollama returned empty/whitespace answer "
                    f"(attempt={attempt['name']} index={attempt_index})"
                )
                attempt_debug["response"]["empty_answer"] = True
            except Exception as exc:
                last_error = exc
                attempt_debug["response"] = {
                    "raw": parsed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            finally:
                elapsed_ms = int(round((time.perf_counter() - started) * 1000))
                attempt_debug["timing_ms"] = elapsed_ms
                if ollama_debug_hook is not None:
                    ollama_debug_hook(attempt_debug)
        raise RuntimeError(
            "Ollama generation failed after retries; see saved attempt request/response artifacts."
        ) from last_error

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
