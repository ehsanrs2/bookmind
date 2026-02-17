"""Bench-only RAG preview helpers built on Qdrant search results."""

from __future__ import annotations

import copy
import json
import time
import urllib.request
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests

from engines.vlm_providers import (
    OllamaProvider,
    OpenAICompatProvider,
    extract_ollama_usage,
    parse_ollama_think,
    parse_ollama_chat_payload,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a technical assistant. Provide concise, factual answers grounded in the "
    "provided document context."
)
OLLAMA_RETRY_SYSTEM_PROMPT = (
    "Do not think aloud. Output JSON only in content. "
    'JSON shape: {"answer":"...","citations":[{"page":1,"stable_id":"..."}]}. '
    "No markdown or extra keys."
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
        "- Return a SINGLE JSON object in the assistant final message content:\n"
        '{"answer":"...","citations":[{"page":1,"stable_id":"..."}]}\n'
        "- No extra keys.\n"
        "- No markdown.\n"
        "- No prose outside JSON.\n"
        "- Keep answer concise.\n"
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


def _extract_query_and_context(messages: Sequence[Dict[str, str]]) -> Tuple[str, str]:
    query = ""
    context = ""
    for message in messages:
        if str(message.get("role") or "") != "user":
            continue
        content = str(message.get("content") or "")
        query_marker = "User query:\n"
        ctx_marker = "Retrieved context:\n```\n"
        if query_marker in content:
            query_start = content.find(query_marker) + len(query_marker)
            query_end = content.find("\n\n", query_start)
            if query_end < 0:
                query_end = len(content)
            query = content[query_start:query_end].strip()
        if ctx_marker in content:
            ctx_start = content.find(ctx_marker) + len(ctx_marker)
            ctx_end = content.find("\n```", ctx_start)
            if ctx_end < 0:
                ctx_end = len(content)
            context = content[ctx_start:ctx_end].strip()
        break
    return query, context


def _build_extractor_messages(
    messages: Sequence[Dict[str, str]],
    *,
    context_limit: int = 1000,
) -> List[Dict[str, str]]:
    query, context = _extract_query_and_context(messages)
    short_context = _truncate_text(context, context_limit)
    user_content = (
        "User query:\n"
        f"{query}\n\n"
        "Retrieved context:\n"
        "```\n"
        f"{short_context}\n"
        "```\n\n"
        "Previous attempt produced internal reasoning but no final answer. "
        "Produce ONLY the JSON object now.\n"
        "Output shape:\n"
        '{"answer":"...","citations":[{"page":1,"stable_id":"..."}]}\n'
        "No extra keys. No markdown. No prose outside JSON."
    )
    return [
        {"role": "system", "content": OLLAMA_RETRY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _response_debug_excerpt(parsed_payload: Dict[str, Any]) -> str:
    raw = parsed_payload.get("raw")
    if isinstance(raw, str):
        text = raw
    else:
        text = _compact_json(raw, limit=2000)
    return _truncate_text(text, 500)


def _build_attempt_failure_error(
    *,
    attempt_name: str,
    attempt_index: int,
    reason: str,
) -> RuntimeError:
    return RuntimeError(
        "Ollama returned invalid assistant JSON content "
        f"(attempt={attempt_name} index={attempt_index}): {reason}"
    )


def _artifact_prefix_for_attempt_name(attempt_name: str) -> str:
    if attempt_name == "initial":
        return "ollama"
    if attempt_name == "retry_strict_json":
        return "retry"
    if attempt_name == "extractor_json_only":
        return "extract"
    return attempt_name


def write_ollama_attempt_artifacts(
    *,
    per_query_dir: Path,
    query_prefix: str,
    attempts: Sequence[Dict[str, Any]],
) -> None:
    if not attempts:
        return
    per_query_dir.mkdir(parents=True, exist_ok=True)
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        attempt_name = str(attempt.get("attempt_name") or "")
        prefix = _artifact_prefix_for_attempt_name(attempt_name)
        request_payload = (
            attempt.get("request")
            if isinstance(attempt.get("request"), dict)
            else {"raw": attempt.get("request")}
        )
        response_payload = attempt if isinstance(attempt, dict) else {"raw": attempt}
        (per_query_dir / f"{query_prefix}_{prefix}_request.json").write_text(
            json.dumps(request_payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (per_query_dir / f"{query_prefix}_{prefix}_response.json").write_text(
            json.dumps(response_payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )


def write_rag_preview_ollama_failure_artifacts(
    *,
    attempts: Sequence[Dict[str, Any]],
) -> Optional[Path]:
    if not attempts:
        return None
    temp_dir = Path(mkdtemp(prefix="bookmind_rag_preview_"))
    write_ollama_attempt_artifacts(
        per_query_dir=temp_dir,
        query_prefix="q01",
        attempts=attempts,
    )
    return temp_dir


def generate_answer(
    provider: Any,
    messages: Sequence[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.2,
    ollama_num_predict: int = 1536,
    ollama_retry_num_predict: int = 2048,
    ollama_think: bool | str = False,
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

    del fallback_citations
    if isinstance(provider, OllamaProvider):
        think_value = parse_ollama_think(ollama_think)
        num_ctx = getattr(provider, "ollama_num_ctx", None)
        debug_hook = ollama_debug_hook

        attempts = [
            {
                "name": "initial",
                "context_limit": None,
                "num_predict": int(ollama_num_predict),
                "temperature": float(temperature),
                "simplify_system_prompt": False,
                "strict_json": False,
            },
            {
                "name": "retry_strict_json",
                "context_limit": 1200,
                "num_predict": int(ollama_retry_num_predict),
                "temperature": 0.0,
                "simplify_system_prompt": True,
                "strict_json": True,
            },
            {
                "name": "extractor_json_only",
                "context_limit": None,
                "num_predict": 512,
                "temperature": 0.0,
                "simplify_system_prompt": True,
                "strict_json": False,
            },
        ]
        last_error: Optional[Exception] = None
        last_failure_fields: Dict[str, Any] = {}
        saw_empty_content_on_retry = False
        for attempt_index, attempt in enumerate(attempts):
            attempt_messages: List[Dict[str, str]]
            if attempt["name"] == "extractor_json_only":
                if not saw_empty_content_on_retry:
                    continue
                attempt_messages = _build_extractor_messages(messages, context_limit=1000)
            else:
                attempt_messages = _prepare_ollama_retry_messages(
                    messages,
                    context_limit=attempt["context_limit"],
                    simplify_system_prompt=attempt["simplify_system_prompt"],
                    strict_json=attempt["strict_json"],
                )
            options: Dict[str, Any] = {
                "num_predict": attempt["num_predict"],
                "temperature": attempt["temperature"],
            }
            if isinstance(num_ctx, int) and num_ctx > 0:
                options["num_ctx"] = int(num_ctx)
            payload: Dict[str, Any] = {
                "model": provider.model,
                "messages": attempt_messages,
                "stream": False,
                "think": think_value,
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
                prompt_eval_count = usage.get("prompt_eval_count")
                eval_count = usage.get("eval_count")
                attempt_debug["response"] = {
                    "raw": parsed_payload.get("raw"),
                    "raw_head_500": _response_debug_excerpt(parsed_payload),
                    "parsed_keys": parsed_payload.get("parsed_keys"),
                    "content_source": parsed_payload.get("source"),
                    "done_reason": parsed_payload.get("done_reason"),
                    "provider_error": parsed_payload.get("error"),
                    "thinking_len_chars": len(thinking),
                    "answer_len_chars": len(cleaned),
                    "prompt_eval_count": prompt_eval_count,
                    "eval_count": eval_count,
                    "usage": usage,
                }
                if cleaned:
                    try:
                        cleaned, _parsed_citations = _extract_json_answer(cleaned)
                    except Exception as exc:  # noqa: BLE001
                        attempt_debug["response"]["json_parse_error"] = str(exc)
                        last_failure_fields = {
                            "failure_mode": "json_parse_error",
                            "done_reason": parsed_payload.get("done_reason"),
                            "eval_count": eval_count,
                            "prompt_eval_count": prompt_eval_count,
                            "raw_response": parsed_payload.get("raw"),
                        }
                        last_error = _build_attempt_failure_error(
                            attempt_name=str(attempt["name"]),
                            attempt_index=attempt_index,
                            reason=str(exc),
                        )
                        continue
                    usage_with_attempt = dict(usage)
                    usage_with_attempt["attempt_index"] = attempt_index
                    usage_with_attempt["attempt_name"] = attempt["name"]
                    return cleaned, usage_with_attempt
                provider_error_text = parsed_payload.get("error")
                if isinstance(provider_error_text, str) and provider_error_text.strip():
                    provider_error_text = provider_error_text.strip()
                    last_failure_fields = {
                        "failure_mode": "provider_error",
                        "done_reason": parsed_payload.get("done_reason"),
                        "eval_count": eval_count,
                        "prompt_eval_count": prompt_eval_count,
                        "raw_response": parsed_payload.get("raw"),
                    }
                    last_error = _build_attempt_failure_error(
                        attempt_name=str(attempt["name"]),
                        attempt_index=attempt_index,
                        reason=f"provider error: {provider_error_text}",
                    )
                    attempt_debug["response"]["provider_error_only"] = True
                    continue
                if thinking:
                    attempt_debug["response"]["empty_content_with_thinking"] = True
                last_failure_fields = {
                    "failure_mode": "empty_content",
                    "done_reason": parsed_payload.get("done_reason"),
                    "eval_count": eval_count,
                    "prompt_eval_count": prompt_eval_count,
                    "raw_response": parsed_payload.get("raw"),
                }
                if attempt["name"] == "retry_strict_json":
                    saw_empty_content_on_retry = True
                last_error = _build_attempt_failure_error(
                    attempt_name=str(attempt["name"]),
                    attempt_index=attempt_index,
                    reason="empty message.content",
                )
                attempt_debug["response"]["empty_answer"] = True
            except Exception as exc:
                last_error = exc
                last_failure_fields = {
                    "failure_mode": "provider_exception",
                    "done_reason": None,
                    "eval_count": None,
                    "prompt_eval_count": None,
                    "raw_response": parsed,
                }
                attempt_debug["response"] = {
                    "raw": parsed,
                    "raw_head_500": _truncate_text(_compact_json(parsed, limit=2000), 500),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            finally:
                elapsed_ms = int(round((time.perf_counter() - started) * 1000))
                attempt_debug["timing_ms"] = elapsed_ms
                attempt_debug["status_code"] = int(
                    getattr(response, "status_code", attempt_debug.get("http_status", 0)) or 0
                )
                if debug_hook is not None:
                    debug_hook(attempt_debug)
        if last_error is not None and last_failure_fields:
            if think_value is False and last_failure_fields.get("failure_mode") == "empty_content":
                raw_head = _truncate_text(
                    _compact_json(last_failure_fields.get("raw_response"), limit=2000),
                    500,
                )
                raise RuntimeError(
                    "Ollama returned empty assistant content with think=false; "
                    f"done_reason={last_failure_fields.get('done_reason')} "
                    f"eval_count={last_failure_fields.get('eval_count')} "
                    f"prompt_eval_count={last_failure_fields.get('prompt_eval_count')} "
                    f"raw_head_500={raw_head}"
                ) from last_error
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
