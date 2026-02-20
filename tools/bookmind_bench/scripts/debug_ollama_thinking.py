#!/usr/bin/env python3
"""Minimal Ollama endpoint reproducer for empty content/response + large thinking."""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


def _load_parser_helpers():
    import sys

    repo_module_root = Path(__file__).resolve().parents[1]
    if str(repo_module_root) not in sys.path:
        sys.path.insert(0, str(repo_module_root))
    from engines.vlm_providers import parse_ollama_chat_payload, parse_ollama_generate_payload

    return parse_ollama_chat_payload, parse_ollama_generate_payload


def _parse_bool(value: object) -> bool:
    cleaned = str(value).strip().lower()
    if cleaned in {"1", "true", "yes", "y"}:
        return True
    if cleaned in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false value, got: {value!r}")


def _build_messages(prompt: str, format_mode: str) -> List[Dict[str, str]]:
    system = "Answer briefly. Do not include internal reasoning."
    base_user = str(prompt or "").strip() or "What is shown in this image?"
    if format_mode == "json-in-text":
        user = (
            f"{base_user}\n\n"
            "Return exactly one JSON object in message.content.\n"
            '{"answer":"...","citations":[{"page":1,"stable_id":"..."}]}\n'
            "No markdown. No extra keys."
        )
    else:
        user = f"{base_user}\nAnswer in one short sentence."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def run_ollama_debug(
    *,
    image: str | None,
    prompt: str,
    api: str = "chat",
    ollama_url: str = "http://127.0.0.1:11434",
    ollama_model: str = "qwen3-vl:latest",
    num_predict: int = 512,
    num_ctx: int | None = None,
    think: bool = False,
    format_mode: str = "json-in-text",
    trials: int = 10,
    out_dir: str = "/tmp/bookmind_ollama_debug",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    parse_ollama_chat_payload, parse_ollama_generate_payload = _load_parser_helpers()
    image_path = Path(image) if image else None
    output_dir = Path(out_dir)
    image_b64 = (
        base64.b64encode(image_path.read_bytes()).decode("ascii")
        if image_path is not None
        else None
    )
    messages = _build_messages(prompt=prompt, format_mode=format_mode)
    api_mode = str(api or "chat").strip().lower()

    counts = {
        "empty_response": 0,
        "empty_content": 0,
        "non_empty_content": 0,
        "with_thinking": 0,
        "empty_content_with_thinking": 0,
        "done_reason_length": 0,
        "eval_count_eq_num_predict": 0,
        "http_error": 0,
    }
    attempts: List[Dict[str, Any]] = []

    for idx in range(1, int(trials) + 1):
        options: Dict[str, Any] = {
            "num_predict": int(num_predict),
            "temperature": float(temperature),
        }
        if isinstance(num_ctx, int) and num_ctx > 0:
            options["num_ctx"] = int(num_ctx)
        if api_mode == "generate":
            prompt_lines = [
                f"System:\n{messages[0]['content']}",
                f"User:\n{messages[1]['content']}",
                "Assistant:",
            ]
            payload = {
                "model": str(ollama_model),
                "prompt": "\n\n".join(prompt_lines),
                "stream": False,
                "think": bool(think),
                "options": options,
            }
            target_endpoint = "/api/generate"
        else:
            user_payload: Dict[str, Any] = {
                "role": "user",
                "content": str(messages[1]["content"]),
            }
            if image_b64:
                user_payload["images"] = [image_b64]
            payload = {
                "model": str(ollama_model),
                "messages": [dict(messages[0]), user_payload],
                "stream": False,
                "think": bool(think),
                "options": options,
            }
            target_endpoint = "/api/chat"
        request_path = output_dir / f"attempt_{idx:02d}_request.json"
        response_path = output_dir / f"attempt_{idx:02d}_response.json"
        _write_json(request_path, payload)

        started = time.perf_counter()
        record: Dict[str, Any] = {
            "attempt": idx,
            "request_file": str(request_path),
            "response_file": str(response_path),
        }
        try:
            response = requests.post(
                str(ollama_url).rstrip("/") + target_endpoint,
                json=payload,
                timeout=120,
            )
            raw_payload: Any
            try:
                raw_payload = response.json()
            except Exception:
                raw_payload = {"raw_text": response.text}
            elapsed_ms = int(round((time.perf_counter() - started) * 1000))
            parsed = (
                parse_ollama_generate_payload(raw_payload)
                if api_mode == "generate"
                else parse_ollama_chat_payload(raw_payload)
            )
            answer = str(parsed.get("answer_text") or "")
            thinking = str(parsed.get("thinking_text") or "")
            if not thinking and isinstance(raw_payload, dict):
                thinking = str(raw_payload.get("thinking") or "")
            done_reason = str(parsed.get("done_reason") or "")
            eval_count = (
                raw_payload.get("eval_count")
                if isinstance(raw_payload, dict)
                else None
            )
            prompt_eval_count = (
                raw_payload.get("prompt_eval_count")
                if isinstance(raw_payload, dict)
                else None
            )
            empty_content = not answer.strip()
            if empty_content:
                counts["empty_response"] += 1
            has_thinking = bool(thinking.strip())
            if empty_content:
                counts["empty_content"] += 1
            else:
                counts["non_empty_content"] += 1
            if has_thinking:
                counts["with_thinking"] += 1
            if empty_content and has_thinking:
                counts["empty_content_with_thinking"] += 1
            if done_reason.lower() == "length":
                counts["done_reason_length"] += 1
            if isinstance(eval_count, int) and eval_count >= int(num_predict):
                counts["eval_count_eq_num_predict"] += 1

            record.update(
                {
                    "status_code": int(response.status_code),
                    "elapsed_ms": elapsed_ms,
                    "empty_content": empty_content,
                    "has_thinking": has_thinking,
                    "done_reason": done_reason or None,
                    "eval_count": eval_count,
                    "prompt_eval_count": prompt_eval_count,
                    "answer_preview": answer.strip()[:200],
                    "thinking_len": len(thinking),
                }
            )
            _write_json(response_path, {"http_status": response.status_code, "response": raw_payload})
        except Exception as exc:  # noqa: BLE001
            counts["http_error"] += 1
            record.update({"error_type": type(exc).__name__, "error": str(exc)})
            _write_json(response_path, {"error_type": type(exc).__name__, "error": str(exc)})
        attempts.append(record)

    summary = {
        "config": {
            "image": str(image_path) if image_path is not None else None,
            "api": api_mode,
            "ollama_url": ollama_url,
            "ollama_model": ollama_model,
            "num_predict": int(num_predict),
            "num_ctx": num_ctx,
            "think": bool(think),
            "format_mode": format_mode,
            "trials": int(trials),
            "temperature": float(temperature),
        },
        "counts": counts,
        "attempts": attempts,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Ollama empty-content/thinking behavior without Qdrant.",
    )
    parser.add_argument(
        "--api",
        choices=["chat", "generate"],
        default="chat",
        help="Ollama endpoint mode (default chat)",
    )
    parser.add_argument(
        "--image",
        required=False,
        help="Optional local image file path (used in chat mode)",
    )
    parser.add_argument("--prompt", default="What does this figure show?", help="User prompt")
    parser.add_argument(
        "--ollama_url",
        default="http://127.0.0.1:11434",
        help="Ollama URL (default http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--ollama_model",
        default="qwen3-vl:latest",
        help="Ollama model tag (default qwen3-vl:latest)",
    )
    parser.add_argument("--num_predict", type=int, default=512, help="options.num_predict")
    parser.add_argument("--num_ctx", type=int, required=False, help="Optional options.num_ctx")
    parser.add_argument("--think", type=_parse_bool, default=False, help="Ollama top-level think")
    parser.add_argument(
        "--format_mode",
        choices=["text", "json-in-text"],
        default="json-in-text",
        help="Prompt format mode (default json-in-text)",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of attempts (default 10)")
    parser.add_argument(
        "--out",
        default="/tmp/bookmind_ollama_debug",
        help="Output directory (default /tmp/bookmind_ollama_debug)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default 0.0)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    summary = run_ollama_debug(
        image=args.image,
        prompt=args.prompt,
        api=args.api,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        think=args.think,
        format_mode=args.format_mode,
        trials=args.trials,
        out_dir=args.out,
        temperature=args.temperature,
    )
    counts = summary.get("counts", {})
    print(f"Saved debug artifacts to {args.out}")
    print(
        "counts "
        f"empty_response={counts.get('empty_response')} "
        f"empty_content={counts.get('empty_content')} "
        f"empty_content_with_thinking={counts.get('empty_content_with_thinking')} "
        f"done_reason_length={counts.get('done_reason_length')} "
        f"eval_count_eq_num_predict={counts.get('eval_count_eq_num_predict')} "
        f"http_error={counts.get('http_error')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
