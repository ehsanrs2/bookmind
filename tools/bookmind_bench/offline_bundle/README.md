# Bookmind Bench Offline Bundle

This directory holds the offline wheelhouse and model caches required by
`tools/bookmind_bench/` on air-gapped machines.

Contents:
- `wheels/` — Python wheels downloaded by `scripts/prefetch_online.sh`.
- `models/marker/` — Marker cache directory.
- `models/paddleocr/` — PaddleOCR/PP-Structure model cache.
- `models/qwen3_vl/` — Hugging Face snapshot for Qwen3-VL.

Cache environment variables used by the scripts:
- Marker: `XDG_CACHE_HOME` and `MARKER_CACHE_DIR` -> `models/marker/`
- PaddleOCR: `PADDLEOCR_HOME` and `PADDLE_HOME` -> `models/paddleocr/`
- Hugging Face: `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_CACHE` -> `models/qwen3_vl/`

For air-gapped vLLM usage, point vLLM to the local model path:
- Example: `python -m vllm.entrypoints.openai.api_server --model tools/bookmind_bench/offline_bundle/models/qwen3_vl`

The bundle contents are intentionally git-ignored. Keep this folder alongside
this repo when transferring to an offline machine.
