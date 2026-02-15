"""Provider abstraction for local VLM caption backends."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    def __init__(self, ollama_url: str, model: str, timeout_s: int = 60) -> None:
        self.ollama_url = ollama_url.rstrip("/")
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
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        start = time.perf_counter()
        response = requests.post(
            self.ollama_url + "/api/chat",
            json=payload,
            timeout=self.timeout_s,
        )
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))
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
            "timing_ms": elapsed_ms,
        }
        return content.strip(), usage


def build_vlm_provider(
    *,
    backend: str,
    endpoint: str,
    model: str,
    ollama_url: str,
    ollama_model: str,
    timeout_s: int = 60,
) -> VLMProvider:
    backend_norm = backend.strip().lower()
    if backend_norm == "ollama":
        return OllamaProvider(ollama_url=ollama_url, model=ollama_model, timeout_s=timeout_s)
    if backend_norm == "vllm":
        return OpenAICompatProvider(endpoint=endpoint, model=model, timeout_s=timeout_s)
    raise ValueError(f"Unsupported VLM backend: {backend}")
