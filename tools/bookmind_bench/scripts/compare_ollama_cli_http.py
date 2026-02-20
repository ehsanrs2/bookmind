#!/usr/bin/env python3
"""Compare Ollama CLI vs HTTP generate/chat for empty-final diagnostics."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import requests


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")


def _head(text: str, limit: int = 200) -> str:
    return str(text or "")[:limit]


def _metric_record(
    *,
    final_text: str,
    thinking_text: Optional[str],
    done_reason: Optional[str],
    eval_count: Optional[int],
    prompt_eval_count: Optional[int],
    error_text: Optional[str],
) -> Dict[str, Any]:
    final_clean = str(final_text or "").strip()
    thinking_clean = str(thinking_text or "").strip()
    return {
        "final_text": final_clean,
        "thinking_text": thinking_clean,
        "final_head_200": _head(final_clean, 200),
        "thinking_head_200": _head(thinking_clean, 200),
        "non_empty_final": bool(final_clean),
        "thinking_present": bool(thinking_clean) if thinking_text is not None else None,
        "done_reason": done_reason,
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "error": error_text,
    }


def extract_chat_metrics(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return _metric_record(
            final_text="",
            thinking_text=None,
            done_reason=None,
            eval_count=None,
            prompt_eval_count=None,
            error_text=f"non-dict payload: {type(payload).__name__}",
        )
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    final_text = message.get("content") if isinstance(message.get("content"), str) else ""
    thinking_text = message.get("thinking") if isinstance(message.get("thinking"), str) else None
    done_reason = payload.get("done_reason") if isinstance(payload.get("done_reason"), str) else None
    eval_count = payload.get("eval_count") if isinstance(payload.get("eval_count"), int) else None
    prompt_eval_count = (
        payload.get("prompt_eval_count") if isinstance(payload.get("prompt_eval_count"), int) else None
    )
    error_text = payload.get("error") if isinstance(payload.get("error"), str) else None
    return _metric_record(
        final_text=final_text,
        thinking_text=thinking_text,
        done_reason=done_reason,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
        error_text=error_text,
    )


def extract_generate_metrics(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return _metric_record(
            final_text="",
            thinking_text=None,
            done_reason=None,
            eval_count=None,
            prompt_eval_count=None,
            error_text=f"non-dict payload: {type(payload).__name__}",
        )
    final_text = payload.get("response") if isinstance(payload.get("response"), str) else ""
    if not final_text:
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        msg_content = message.get("content")
        if isinstance(msg_content, str):
            final_text = msg_content
    thinking_text: Optional[str] = None
    if isinstance(payload.get("thinking"), str):
        thinking_text = payload.get("thinking")
    else:
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        if isinstance(message.get("thinking"), str):
            thinking_text = message.get("thinking")
    done_reason = payload.get("done_reason") if isinstance(payload.get("done_reason"), str) else None
    eval_count = payload.get("eval_count") if isinstance(payload.get("eval_count"), int) else None
    prompt_eval_count = (
        payload.get("prompt_eval_count") if isinstance(payload.get("prompt_eval_count"), int) else None
    )
    error_text = payload.get("error") if isinstance(payload.get("error"), str) else None
    return _metric_record(
        final_text=final_text,
        thinking_text=thinking_text,
        done_reason=done_reason,
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
        error_text=error_text,
    )


def summarize_mode(attempts: List[Dict[str, Any]], num_predict: int) -> Dict[str, Any]:
    timings = [float(a.get("timing_ms") or 0.0) for a in attempts]
    eval_present = [a for a in attempts if isinstance(a.get("eval_count"), int)]
    return {
        "trials_total": len(attempts),
        "empty_final_count": sum(1 for a in attempts if not bool(a.get("non_empty_final"))),
        "empty_final_with_thinking_count": sum(
            1
            for a in attempts
            if (not bool(a.get("non_empty_final"))) and bool(a.get("thinking_present"))
        ),
        "done_reason_length_count": sum(
            1 for a in attempts if str(a.get("done_reason") or "").strip().lower() == "length"
        ),
        "eval_count_eq_num_predict_count": sum(
            1 for a in eval_present if int(a.get("eval_count")) == int(num_predict)
        ),
        "avg_timing_ms": round(mean(timings), 3) if timings else 0.0,
    }


def _options_payload(num_predict: int, temperature: float, num_ctx: Optional[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "num_predict": int(num_predict),
        "temperature": float(temperature),
    }
    if isinstance(num_ctx, int):
        out["num_ctx"] = int(num_ctx)
    return out


def _render_report(
    *,
    out_dir: Path,
    config: Dict[str, Any],
    attempts_by_mode: Dict[str, List[Dict[str, Any]]],
    summary_by_mode: Dict[str, Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Ollama CLI vs HTTP Comparison")
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(config, ensure_ascii=True, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Mode Summaries")
    for mode in ("cli", "generate", "chat"):
        s = summary_by_mode.get(mode, {})
        lines.append(f"### {mode}")
        lines.append(f"- trials_total: {s.get('trials_total')}")
        lines.append(f"- empty_final_count: {s.get('empty_final_count')}")
        lines.append(f"- empty_final_with_thinking_count: {s.get('empty_final_with_thinking_count')}")
        lines.append(f"- done_reason_length_count: {s.get('done_reason_length_count')}")
        lines.append(
            f"- eval_count_eq_num_predict_count: {s.get('eval_count_eq_num_predict_count')}"
        )
        lines.append(f"- avg_timing_ms: {s.get('avg_timing_ms')}")
        lines.append("")
    lines.append("## Trial Excerpts")
    for mode in ("cli", "generate", "chat"):
        lines.append(f"### {mode}")
        for row in attempts_by_mode.get(mode, []):
            lines.append(
                f"- attempt {row.get('attempt'):02d}: "
                f"non_empty_final={row.get('non_empty_final')} "
                f"thinking_present={row.get('thinking_present')} "
                f"done_reason={row.get('done_reason')} "
                f"eval_count={row.get('eval_count')} "
                f"prompt_eval_count={row.get('prompt_eval_count')}"
            )
            lines.append(f"  final_head_200: {row.get('final_head_200')}")
            lines.append(f"  thinking_head_200: {row.get('thinking_head_200')}")
        lines.append("")

    suspicious: Optional[Dict[str, Any]] = None
    suspicious_mode = ""
    for mode in ("generate", "chat", "cli"):
        for row in attempts_by_mode.get(mode, []):
            if (
                not bool(row.get("non_empty_final"))
                and bool(row.get("thinking_present"))
                and str(row.get("done_reason") or "").strip().lower() == "length"
            ):
                suspicious = row
                suspicious_mode = mode
                break
        if suspicious is not None:
            break
    lines.append("## Most Suspicious Case")
    if suspicious is None:
        lines.append("- none")
    else:
        lines.append(
            f"- mode={suspicious_mode} attempt={suspicious.get('attempt')} "
            f"done_reason={suspicious.get('done_reason')} "
            f"eval_count={suspicious.get('eval_count')} "
            f"prompt_eval_count={suspicious.get('prompt_eval_count')}"
        )
        lines.append(f"- final_head_200: {suspicious.get('final_head_200')}")
        lines.append(f"- thinking_head_200: {suspicious.get('thinking_head_200')}")
    lines.append("")
    _write_text(out_dir / "compare.md", "\n".join(lines))


def run_compare(
    *,
    ollama_url: str,
    model: str,
    image: str,
    prompt: str,
    num_predict: int,
    num_ctx: Optional[int],
    temperature: float,
    trials: int,
    out: str,
) -> Dict[str, Any]:
    out_dir = Path(out)
    image_path = Path(image).resolve()
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    opts = _options_payload(num_predict=num_predict, temperature=temperature, num_ctx=num_ctx)
    attempts_by_mode: Dict[str, List[Dict[str, Any]]] = {"cli": [], "generate": [], "chat": []}

    for idx in range(1, int(trials) + 1):
        # CLI
        cli_request = {
            "prompt": prompt,
            "images": [str(image_path)],
            "options": opts,
        }
        _write_json(out_dir / "cli" / f"attempt_{idx:02d}_request.json", cli_request)
        cli_started = time.perf_counter()
        try:
            run = subprocess.run(
                ["ollama", "run", str(model)],
                input=json.dumps(cli_request, ensure_ascii=True),
                text=True,
                capture_output=True,
                check=False,
            )
            cli_timing = int(round((time.perf_counter() - cli_started) * 1000))
            _write_text(out_dir / "cli" / f"attempt_{idx:02d}_stdout.txt", run.stdout)
            _write_text(out_dir / "cli" / f"attempt_{idx:02d}_stderr.txt", run.stderr)
            _write_json(
                out_dir / "cli" / f"attempt_{idx:02d}_meta.json",
                {"exit_code": int(run.returncode), "timing_ms": cli_timing},
            )
            cli_metrics = _metric_record(
                final_text=run.stdout,
                thinking_text=None,
                done_reason=None,
                eval_count=None,
                prompt_eval_count=None,
                error_text=run.stderr.strip() or None,
            )
            cli_metrics["attempt"] = idx
            cli_metrics["timing_ms"] = cli_timing
            attempts_by_mode["cli"].append(cli_metrics)
        except Exception as exc:  # noqa: BLE001
            cli_timing = int(round((time.perf_counter() - cli_started) * 1000))
            _write_text(out_dir / "cli" / f"attempt_{idx:02d}_stdout.txt", "")
            _write_text(out_dir / "cli" / f"attempt_{idx:02d}_stderr.txt", str(exc))
            _write_json(
                out_dir / "cli" / f"attempt_{idx:02d}_meta.json",
                {"exit_code": -1, "timing_ms": cli_timing, "error": str(exc)},
            )
            cli_metrics = _metric_record(
                final_text="",
                thinking_text=None,
                done_reason=None,
                eval_count=None,
                prompt_eval_count=None,
                error_text=str(exc),
            )
            cli_metrics["attempt"] = idx
            cli_metrics["timing_ms"] = cli_timing
            attempts_by_mode["cli"].append(cli_metrics)

        # HTTP generate
        gen_request = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": opts,
        }
        _write_json(out_dir / "generate" / f"attempt_{idx:02d}_request.json", gen_request)
        gen_started = time.perf_counter()
        gen_status = None
        try:
            gen_resp = requests.post(
                str(ollama_url).rstrip("/") + "/api/generate", json=gen_request, timeout=180
            )
            gen_timing = int(round((time.perf_counter() - gen_started) * 1000))
            gen_status = int(gen_resp.status_code)
            try:
                gen_payload: Any = gen_resp.json()
            except Exception:
                gen_payload = {"raw_text": gen_resp.text}
            _write_json(out_dir / "generate" / f"attempt_{idx:02d}_response.json", {"response": gen_payload})
            _write_json(
                out_dir / "generate" / f"attempt_{idx:02d}_meta.json",
                {"status_code": gen_status, "timing_ms": gen_timing},
            )
            gen_metrics = extract_generate_metrics(gen_payload)
            gen_metrics["attempt"] = idx
            gen_metrics["timing_ms"] = gen_timing
            gen_metrics["status_code"] = gen_status
            attempts_by_mode["generate"].append(gen_metrics)
        except Exception as exc:  # noqa: BLE001
            gen_timing = int(round((time.perf_counter() - gen_started) * 1000))
            _write_json(
                out_dir / "generate" / f"attempt_{idx:02d}_response.json",
                {"error": str(exc)},
            )
            _write_json(
                out_dir / "generate" / f"attempt_{idx:02d}_meta.json",
                {"status_code": gen_status, "timing_ms": gen_timing, "error": str(exc)},
            )
            gen_metrics = _metric_record(
                final_text="",
                thinking_text=None,
                done_reason=None,
                eval_count=None,
                prompt_eval_count=None,
                error_text=str(exc),
            )
            gen_metrics["attempt"] = idx
            gen_metrics["timing_ms"] = gen_timing
            gen_metrics["status_code"] = gen_status
            attempts_by_mode["generate"].append(gen_metrics)

        # HTTP chat
        chat_request = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "options": opts,
        }
        _write_json(out_dir / "chat" / f"attempt_{idx:02d}_request.json", chat_request)
        chat_started = time.perf_counter()
        chat_status = None
        try:
            chat_resp = requests.post(
                str(ollama_url).rstrip("/") + "/api/chat", json=chat_request, timeout=180
            )
            chat_timing = int(round((time.perf_counter() - chat_started) * 1000))
            chat_status = int(chat_resp.status_code)
            try:
                chat_payload: Any = chat_resp.json()
            except Exception:
                chat_payload = {"raw_text": chat_resp.text}
            _write_json(out_dir / "chat" / f"attempt_{idx:02d}_response.json", {"response": chat_payload})
            _write_json(
                out_dir / "chat" / f"attempt_{idx:02d}_meta.json",
                {"status_code": chat_status, "timing_ms": chat_timing},
            )
            chat_metrics = extract_chat_metrics(chat_payload)
            chat_metrics["attempt"] = idx
            chat_metrics["timing_ms"] = chat_timing
            chat_metrics["status_code"] = chat_status
            attempts_by_mode["chat"].append(chat_metrics)
        except Exception as exc:  # noqa: BLE001
            chat_timing = int(round((time.perf_counter() - chat_started) * 1000))
            _write_json(
                out_dir / "chat" / f"attempt_{idx:02d}_response.json",
                {"error": str(exc)},
            )
            _write_json(
                out_dir / "chat" / f"attempt_{idx:02d}_meta.json",
                {"status_code": chat_status, "timing_ms": chat_timing, "error": str(exc)},
            )
            chat_metrics = _metric_record(
                final_text="",
                thinking_text=None,
                done_reason=None,
                eval_count=None,
                prompt_eval_count=None,
                error_text=str(exc),
            )
            chat_metrics["attempt"] = idx
            chat_metrics["timing_ms"] = chat_timing
            chat_metrics["status_code"] = chat_status
            attempts_by_mode["chat"].append(chat_metrics)

    summary_by_mode = {
        mode: summarize_mode(rows, num_predict=num_predict)
        for mode, rows in attempts_by_mode.items()
    }
    config = {
        "ollama_url": ollama_url,
        "model": model,
        "image": str(image_path),
        "prompt": prompt,
        "num_predict": int(num_predict),
        "num_ctx": num_ctx,
        "temperature": float(temperature),
        "trials": int(trials),
        "out": str(out_dir),
    }
    payload = {
        "config": config,
        "summary_by_mode": summary_by_mode,
        "attempts_by_mode": attempts_by_mode,
    }
    _write_json(out_dir / "compare_summary.json", payload)
    _render_report(
        out_dir=out_dir,
        config=config,
        attempts_by_mode=attempts_by_mode,
        summary_by_mode=summary_by_mode,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Ollama CLI vs HTTP chat/generate outputs.")
    parser.add_argument("--ollama_url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3-vl:latest")
    parser.add_argument(
        "--image",
        required=False,
        default="tools/bookmind_bench/tests/fixtures/figure3_tiny.png",
        help="Path to local image",
    )
    parser.add_argument(
        "--prompt",
        default="What does this figure show? Answer in 1-2 sentences.",
    )
    parser.add_argument("--num_predict", type=int, default=256)
    parser.add_argument("--num_ctx", type=int, required=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", default="/tmp/bookmind_ollama_compare")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_compare(
        ollama_url=args.ollama_url,
        model=args.model,
        image=args.image,
        prompt=args.prompt,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        temperature=args.temperature,
        trials=args.trials,
        out=args.out,
    )
    print(f"Artifacts: {args.out}")
    for mode in ("cli", "generate", "chat"):
        summary = result["summary_by_mode"][mode]
        print(
            f"{mode}: empty_final={summary['empty_final_count']} "
            f"empty_final_with_thinking={summary['empty_final_with_thinking_count']} "
            f"done_reason_length={summary['done_reason_length_count']} "
            f"eval_eq_num_predict={summary['eval_count_eq_num_predict_count']} "
            f"avg_timing_ms={summary['avg_timing_ms']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
