"""Bench-only RAG preview helpers built on Qdrant search results."""

from __future__ import annotations

import copy
import json
import re
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
    "Return ONLY the JSON object. Do not include analysis or thoughts. Output JSON immediately. "
    'JSON shape: {"answer":"...","citations":[{"page":1,"stable_id":"..."}]}. '
    "No markdown or extra keys."
)
_THINKING_PREFERRED_MARKERS = (
    "answer should be",
    "final answer",
    "in summary",
    "therefore",
    "figure",
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


def _rag_user_instructions_json() -> str:
    return (
        "Instructions:\n"
        "- Answer using only the provided context.\n"
        "- If the context is insufficient, explicitly say context is insufficient.\n"
        "- Return ONLY valid JSON in this exact shape:\n"
        '{"answer":"...","citations":[{"page":1,"stable_id":"..."}]}\n'
        "- Keep answer concise.\n"
        "- Do not use markdown code fences.\n"
        "- citations may be an empty list."
    )


def _rag_user_instructions_text() -> str:
    return (
        "Instructions:\n"
        "- Answer using only the provided context.\n"
        "- If the context is insufficient, explicitly say the context is insufficient.\n"
        "- Include citations in the form [page:stable_id]."
    )


def build_rag_messages(
    query: str,
    context_text: str,
    *,
    require_json_response: bool = False,
) -> List[Dict[str, str]]:
    user_content = (
        "User query:\n"
        f"{query.strip()}\n\n"
        "Retrieved context:\n"
        "```\n"
        f"{context_text.strip()}\n"
        "```\n\n"
        f"{_rag_user_instructions_json() if require_json_response else _rag_user_instructions_text()}"
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
    strict_json: bool = False,
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
    if strict_json:
        for msg in prepared:
            if str(msg.get("role") or "") == "user":
                content = str(msg.get("content") or "").rstrip()
                strict_tail = (
                    "\n\nSTRICT OUTPUT RULES:\n"
                    "- Return exactly one JSON object.\n"
                    "- Object shape must be "
                    '{"answer":"...","citations":[{"page":1,"stable_id":"..."}]}.\n'
                    "- No markdown, no commentary, no chain-of-thought."
                )
                msg["content"] = content + strict_tail
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


def _extract_first_json_object_text(raw: str) -> str:
    text = str(raw or "")
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object start '{' found in message.content")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
            continue
    raise ValueError("Could not find balanced JSON object in message.content")


def _extract_json_answer(content_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    raw = str(content_text or "").strip()
    if not raw:
        raise ValueError("Ollama returned empty message.content")
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = json.loads(_extract_first_json_object_text(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Ollama response must contain a JSON object")
    answer_value = parsed.get("answer")
    if not isinstance(answer_value, str):
        raise ValueError("Ollama response missing string field 'answer'")
    cleaned = answer_value.strip()
    if not cleaned:
        raise ValueError("Ollama response contains empty 'answer'")
    citations = parsed.get("citations", [])
    if citations is None:
        citations = []
    if not isinstance(citations, list):
        raise ValueError("Ollama response field 'citations' must be a list")
    normalized_citations: List[Dict[str, Any]] = []
    for idx, citation in enumerate(citations):
        if not isinstance(citation, dict):
            raise ValueError(f"Ollama citation at index {idx} must be an object")
        page = citation.get("page")
        stable_id = citation.get("stable_id")
        if not isinstance(page, int) or isinstance(page, bool):
            raise ValueError(f"Ollama citation at index {idx} missing integer 'page'")
        if not isinstance(stable_id, str):
            raise ValueError(f"Ollama citation at index {idx} missing string 'stable_id'")
        normalized_citations.append({"page": page, "stable_id": stable_id})
    return cleaned, normalized_citations


def _compact_json(value: Any, limit: int = 2000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _extract_stable_ids_from_thinking(thinking_text: str) -> List[str]:
    text = str(thinking_text or "")
    ids: List[str] = []
    seen: set[str] = set()
    patterns = (
        re.compile(r"stable[_\s-]*id\s*[:=]\s*([A-Za-z0-9_.:-]+)", re.IGNORECASE),
        re.compile(r"\[\s*\d+\s*:\s*([^\]\s]+)\s*\]"),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            stable_id = str(match.group(1) or "").strip().strip(",.;")
            if stable_id and stable_id not in seen:
                seen.add(stable_id)
                ids.append(stable_id)
    return ids


def _select_fallback_citations(
    thinking_text: str,
    fallback_citations: Sequence[Dict[str, Any]],
    max_items: int = 3,
) -> List[Dict[str, Any]]:
    cleaned_rows: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in fallback_citations:
        if not isinstance(row, dict):
            continue
        stable_id = str(row.get("stable_id") or "").strip()
        page = row.get("page")
        if not stable_id or not isinstance(page, int):
            continue
        cleaned = {"stable_id": stable_id, "page": page}
        cleaned_rows.append(cleaned)
        if stable_id not in by_id:
            by_id[stable_id] = cleaned

    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for stable_id in _extract_stable_ids_from_thinking(thinking_text):
        row = by_id.get(stable_id)
        if row is None:
            continue
        if row["stable_id"] in seen:
            continue
        selected.append(row)
        seen.add(row["stable_id"])
        if len(selected) >= max_items:
            return selected

    for row in cleaned_rows:
        if row["stable_id"] in seen:
            continue
        selected.append(row)
        seen.add(row["stable_id"])
        if len(selected) >= max_items:
            break
    return selected


def _extract_fallback_answer_sentence(thinking_text: str) -> str:
    text = str(thinking_text or "").strip()
    if not text:
        return ""
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    preferred = [
        line for line in lines if any(marker in line.lower() for marker in _THINKING_PREFERRED_MARKERS)
    ]
    candidates = preferred if preferred else lines
    for candidate in candidates:
        if len(candidate) >= 24:
            return _truncate_text(candidate, 300)
    sentence_chunks = re.split(r"(?<=[.!?])\s+", text)
    sentence_chunks = [chunk.strip() for chunk in sentence_chunks if chunk and chunk.strip()]
    if sentence_chunks:
        return _truncate_text(sentence_chunks[-1], 300)
    return _truncate_text(text, 300)


def _build_answer_from_thinking_fallback(
    thinking_text: str,
    fallback_citations: Sequence[Dict[str, Any]],
) -> str:
    base = _extract_fallback_answer_sentence(thinking_text)
    if not base:
        return ""
    selected = _select_fallback_citations(thinking_text, fallback_citations)
    if not selected:
        return base
    markers = " ".join(f"[{row['page']}:{row['stable_id']}]" for row in selected)
    if re.search(r"\[\s*\d+\s*:[^\]]+\]", base):
        return base
    return f"{base} {markers}".strip()


def generate_answer(
    provider: Any,
    messages: Sequence[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.2,
    ollama_num_predict: int = 1536,
    ollama_retry_num_predict: int = 2048,
    fallback_citations: Optional[Sequence[Dict[str, Any]]] = None,
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
        num_ctx = getattr(provider, "ollama_num_ctx", None)
        debug_hook = ollama_debug_hook
        should_retry = False

        attempts = [
            {
                "name": "initial",
                "context_limit": None,
                "max_tokens": int(ollama_num_predict),
                "temperature": float(temperature),
                "simplify_system_prompt": True,
                "strict_json": True,
            },
            {
                "name": "retry_strict_json",
                "context_limit": 1200,
                "max_tokens": int(ollama_retry_num_predict),
                "temperature": 0.0,
                "simplify_system_prompt": True,
                "strict_json": True,
            },
        ]
        last_error: Optional[Exception] = None
        last_failure_fields: Dict[str, Any] = {}
        for attempt_index, attempt in enumerate(attempts):
            if attempt_index == 1 and not should_retry:
                continue
            attempt_messages = _prepare_ollama_retry_messages(
                messages,
                context_limit=attempt["context_limit"],
                simplify_system_prompt=attempt["simplify_system_prompt"],
                strict_json=attempt["strict_json"],
            )
            options: Dict[str, Any] = {
                "num_predict": attempt["max_tokens"],
                "temperature": attempt["temperature"],
            }
            if isinstance(num_ctx, int) and num_ctx > 0:
                options["num_ctx"] = int(num_ctx)
            payload: Dict[str, Any] = {
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
                attempt_debug["http_status"] = int(getattr(response, "status_code", 200))
                attempt_debug["response_headers"] = _ollama_header_subset(
                    getattr(response, "headers", {})
                )
                response.raise_for_status()
                parsed = _decode_ollama_response(response)
                parsed_payload = parse_ollama_chat_payload(parsed)
                content = parsed_payload.get("answer_text")
                thinking = str(parsed_payload.get("thinking_text") or "").strip()
                cleaned = str(content or "").strip()
                usage = extract_ollama_usage(parsed)
                attempt_debug["response"] = {
                    "raw": parsed_payload.get("raw"),
                    "parsed_keys": parsed_payload.get("parsed_keys"),
                    "content_source": parsed_payload.get("source"),
                    "done_reason": parsed_payload.get("done_reason"),
                    "provider_error": parsed_payload.get("error"),
                    "thinking_len_chars": len(thinking),
                    "answer_len_chars": len(cleaned),
                    "usage": usage,
                }
                if cleaned:
                    try:
                        cleaned, _parsed_citations = _extract_json_answer(cleaned)
                    except Exception as exc:  # noqa: BLE001
                        attempt_debug["response"]["json_parse_error"] = str(exc)
                        last_failure_fields = {
                            "done_reason": parsed_payload.get("done_reason"),
                            "eval_count": usage.get("eval_count"),
                            "prompt_eval_count": usage.get("prompt_eval_count"),
                            "raw_response": parsed_payload.get("raw"),
                        }
                        if attempt_index == 0:
                            should_retry = True
                            last_error = RuntimeError(
                                "Ollama returned non-parseable JSON in message.content "
                                f"(attempt={attempt['name']} index={attempt_index})"
                            )
                            continue
                        last_error = RuntimeError(
                            "Ollama returned non-parseable JSON in message.content "
                            f"(attempt={attempt['name']} index={attempt_index}): {exc}"
                        )
                        continue
                    usage_with_attempt = dict(usage)
                    usage_with_attempt["attempt_index"] = attempt_index
                    usage_with_attempt["attempt_name"] = attempt["name"]
                    return cleaned, usage_with_attempt
                if thinking:
                    attempt_debug["response"]["empty_content_with_thinking"] = True
                last_failure_fields = {
                    "done_reason": parsed_payload.get("done_reason"),
                    "eval_count": usage.get("eval_count"),
                    "prompt_eval_count": usage.get("prompt_eval_count"),
                    "raw_response": parsed_payload.get("raw"),
                }
                if attempt_index == 0:
                    should_retry = True
                    last_error = RuntimeError(
                        "Ollama returned empty message.content "
                        f"(attempt={attempt['name']} index={attempt_index})"
                    )
                    continue
                last_error = RuntimeError(
                    "Ollama returned empty message.content after retry "
                    f"(attempt={attempt['name']} index={attempt_index})"
                )
                attempt_debug["response"]["empty_answer"] = True
                if thinking:
                    fallback_answer = _build_answer_from_thinking_fallback(
                        thinking,
                        fallback_citations or [],
                    )
                    if fallback_answer:
                        usage_with_attempt = dict(usage)
                        usage_with_attempt["attempt_index"] = attempt_index
                        usage_with_attempt["attempt_name"] = attempt["name"]
                        usage_with_attempt["fallback_from_thinking"] = True
                        attempt_debug["response"]["used_thinking_fallback"] = True
                        attempt_debug["response"]["fallback_answer_len_chars"] = len(fallback_answer)
                        return fallback_answer, usage_with_attempt
            except Exception as exc:
                last_error = exc
                last_failure_fields = {
                    "done_reason": None,
                    "eval_count": None,
                    "prompt_eval_count": None,
                    "raw_response": parsed,
                }
                attempt_debug["response"] = {
                    "raw": parsed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            finally:
                elapsed_ms = int(round((time.perf_counter() - started) * 1000))
                attempt_debug["timing_ms"] = elapsed_ms
                if debug_hook is not None:
                    debug_hook(attempt_debug)
        if last_error is not None and last_failure_fields:
            raise RuntimeError(
                "Ollama generation failed after retry; "
                f"done_reason={last_failure_fields.get('done_reason')} "
                f"eval_count={last_failure_fields.get('eval_count')} "
                f"prompt_eval_count={last_failure_fields.get('prompt_eval_count')} "
                f"raw_response={_compact_json(last_failure_fields.get('raw_response'))}"
            ) from last_error
        raise RuntimeError(
            "Ollama generation failed after retry; see saved request/response artifacts."
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
