from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


def _load_compare_runner():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_ollama_cli_http.py"
    )
    spec = importlib.util.spec_from_file_location("bookmind_ollama_compare_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load compare script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_compare


def test_ollama_cli_http_parity_non_empty_final() -> None:
    if os.getenv("BOOKMIND_OLLAMA_INTEGRATION") != "1":
        pytest.skip("Set BOOKMIND_OLLAMA_INTEGRATION=1 to run local Ollama integration checks.")

    fixture = Path(__file__).resolve().parent / "fixtures" / "figure3_tiny.png"
    if not fixture.exists():
        pytest.skip(f"Fixture missing: {fixture}")

    out_dir = Path("/tmp/bookmind_ollama_debug_test")
    run_compare = _load_compare_runner()
    report = run_compare(
        ollama_url="http://127.0.0.1:11434",
        model="qwen3-vl:latest",
        image=str(fixture),
        prompt="What does this figure show? Answer in 1-2 sentences.",
        num_predict=256,
        num_ctx=None,
        temperature=0.0,
        think=False,
        trials=1,
        out=str(out_dir),
    )

    for mode in ("generate", "chat"):
        attempt = report["attempts_by_mode"][mode][0]
        if attempt.get("non_empty_final"):
            continue
        diagnostics = (
            f"mode={mode} non_empty_final={attempt.get('non_empty_final')} "
            f"thinking_present={attempt.get('thinking_present')} "
            f"done_reason={attempt.get('done_reason')} "
            f"eval_count={attempt.get('eval_count')} "
            f"prompt_eval_count={attempt.get('prompt_eval_count')} "
            f"error={attempt.get('error')} artifacts={out_dir}"
        )
        pytest.fail(
            "Ollama HTTP final text was empty. "
            "If this is empty+thinking+length, inspect saved artifacts. "
            + diagnostics
        )
