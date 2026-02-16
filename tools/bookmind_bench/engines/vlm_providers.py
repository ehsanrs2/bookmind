"""Provider abstraction for local VLM caption backends."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


class VLMProvider(ABC):
    """Thin backend abstraction for VLM caption calls."""

    @abstractmethod
    def caption(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Return generated caption text and optional usage metadata."""


def _encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _extract_openai_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("VLM response missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("VLM response missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("VLM response missing content")
    return content.strip()


def _parse_json_maybe(text: str) -> Optional[Any]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _parse_ollama_stream_chunks(chunks: List[Any]) -> Dict[str, Any]:
    fragments: List[str] = []
    parsed_keys: List[str] = []
    last_done_reason: Optional[str] = None
    last_error: Optional[str] = None

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for key in chunk.keys():
            key_s = str(key)
            if key_s not in parsed_keys:
                parsed_keys.append(key_s)
        chunk_error = chunk.get("error")
        if isinstance(chunk_error, str) and chunk_error.strip():
            last_error = chunk_error.strip()

        content: Optional[str] = None
        message = chunk.get("message")
        if isinstance(message, dict):
            message_content = message.get("content")
            if isinstance(message_content, str):
                content = message_content
        if content is None:
            response_text = chunk.get("response")
            if isinstance(response_text, str):
                content = response_text
        if isinstance(content, str):
            fragments.append(content)

        done_reason = chunk.get("done_reason")
        if isinstance(done_reason, str) and done_reason.strip():
            last_done_reason = done_reason.strip()

    answer_text = "".join(fragments)
    return {
        "answer_text": answer_text,
        "source": "stream_chunks",
        "done_reason": last_done_reason,
        "error": last_error,
        "parsed_keys": sorted(parsed_keys),
        "raw": chunks,
    }


def parse_ollama_chat_payload(payload: Any) -> Dict[str, Any]:
    """Parse assistant content from Ollama `/api/chat` payload variants."""
    parsed_keys: List[str] = []
    answer_text: Optional[str] = None
    source: Optional[str] = None
    done_reason: Optional[str] = None
    error_text: Optional[str] = None
    thinking_text: Optional[str] = None

    if isinstance(payload, dict):
        parsed_keys = sorted(str(key) for key in payload.keys())
        error_value = payload.get("error")
        if isinstance(error_value, str) and error_value.strip():
            error_text = error_value.strip()
        done_value = payload.get("done_reason")
        if isinstance(done_value, str) and done_value.strip():
            done_reason = done_value.strip()

        message = payload.get("message")
        if isinstance(message, dict):
            message_content = message.get("content")
            if isinstance(message_content, str):
                answer_text = message_content
                source = "message.content"
            message_thinking = message.get("thinking")
            if isinstance(message_thinking, str):
                thinking_text = message_thinking

        if answer_text is None:
            response_text = payload.get("response")
            if isinstance(response_text, str):
                answer_text = response_text
                source = "response"

        if answer_text is None:
            text_field = payload.get("text")
            if isinstance(text_field, str):
                answer_text = text_field
                source = "text"

        return {
            "answer_text": answer_text,
            "source": source,
            "done_reason": done_reason,
            "error": error_text,
            "thinking_text": thinking_text,
            "parsed_keys": parsed_keys,
            "raw": payload,
        }

    if isinstance(payload, list):
        return _parse_ollama_stream_chunks(payload)

    payload_json = _parse_json_maybe(str(payload or ""))
    if isinstance(payload_json, list):
        return _parse_ollama_stream_chunks(payload_json)
    if isinstance(payload_json, dict):
        return parse_ollama_chat_payload(payload_json)

    chunks: List[Any] = []
    for line in str(payload or "").splitlines():
        line_parsed = _parse_json_maybe(line)
        if line_parsed is not None:
            chunks.append(line_parsed)
    if chunks:
        if len(chunks) == 1 and isinstance(chunks[0], dict):
            return parse_ollama_chat_payload(chunks[0])
        return _parse_ollama_stream_chunks(chunks)

    return {
        "answer_text": None,
        "source": None,
        "done_reason": None,
        "error": None,
        "thinking_text": None,
        "parsed_keys": [],
        "raw": payload,
    }


def extract_ollama_usage(parsed: Any, timing_ms: Optional[int] = None) -> Dict[str, Any]:
    usage: Dict[str, Any] = {}
    if isinstance(parsed, dict):
        usage = {
            "prompt_eval_count": parsed.get("prompt_eval_count"),
            "eval_count": parsed.get("eval_count"),
            "prompt_eval_duration": parsed.get("prompt_eval_duration"),
            "eval_duration": parsed.get("eval_duration"),
            "total_duration": parsed.get("total_duration"),
        }
    if timing_ms is not None:
        usage["timing_ms"] = timing_ms
    return usage


class OpenAICompatProvider(VLMProvider):
    def __init__(self, endpoint: str, model: str, timeout_s: int = 60) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def caption(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        image_b64 = _encode_image_base64(Path(image_path))
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = self.endpoint + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        text = _extract_openai_content(parsed)
        usage = parsed.get("usage")
        if not isinstance(usage, dict):
            usage = None
        return text, usage


class OllamaProvider(VLMProvider):
    def __init__(
        self,
        ollama_url: str,
        model: str,
        timeout_s: int = 60,
        ollama_format: str = "text",
        ollama_num_ctx: Optional[int] = None,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.ollama_format = ollama_format.strip().lower() if ollama_format else "text"
        self.ollama_num_ctx = ollama_num_ctx

    def caption(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        image_b64 = _encode_image_base64(Path(image_path))
        options: Dict[str, Any] = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        if isinstance(self.ollama_num_ctx, int) and self.ollama_num_ctx > 0:
            options["num_ctx"] = int(self.ollama_num_ctx)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "options": options,
        }
        if self.ollama_format == "json":
            payload["format"] = "json"
        start = time.perf_counter()
        response = requests.post(
            self.ollama_url + "/api/chat",
            json=payload,
            timeout=self.timeout_s,
        )
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))
        response.raise_for_status()
        parsed = response.json()
        parsed_content = parse_ollama_chat_payload(parsed)
        content = parsed_content.get("answer_text")
        if content is None:
            keys = parsed_content.get("parsed_keys")
            reason = parsed_content.get("done_reason")
            provider_error = parsed_content.get("error")
            raise ValueError(
                "Ollama response missing assistant content "
                "(expected message.content or response). "
                f"response_keys={keys} done_reason={reason} error={provider_error}"
            )
        usage = extract_ollama_usage(parsed, timing_ms=elapsed_ms)
        return content.strip(), usage


def build_vlm_provider(
    *,
    backend: str,
    endpoint: str,
    model: str,
    ollama_url: str,
    ollama_model: str,
    timeout_s: int = 60,
    ollama_format: str = "text",
    ollama_num_ctx: Optional[int] = None,
) -> VLMProvider:
    backend_norm = backend.strip().lower()
    if backend_norm == "ollama":
        return OllamaProvider(
            ollama_url=ollama_url,
            model=ollama_model,
            timeout_s=timeout_s,
            ollama_format=ollama_format,
            ollama_num_ctx=ollama_num_ctx,
        )
    if backend_norm == "vllm":
        return OpenAICompatProvider(endpoint=endpoint, model=model, timeout_s=timeout_s)
    raise ValueError(f"Unsupported VLM backend: {backend}")
