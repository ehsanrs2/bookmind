# Bookmind Bench Offline Bundle

This directory holds the offline wheelhouse and model caches required by
`tools/bookmind_bench/` on air-gapped machines.

Contents:
- `wheels/` — Python wheels downloaded by `scripts/prefetch_online.sh` (base + per-profile).
- `models/marker/` — Marker cache directory.
- `models/paddleocr/` — PaddleOCR/PP-Structure model cache.
- `models/layoutparser_publaynet/` — LayoutParser/Detectron2 PubLayNet config + weights cache.
- `models/qwen3_vl/4b` — Qwen3-VL 4B snapshot (default).
- `models/qwen3_vl/8b_fp8` — Qwen3-VL 8B FP8 snapshot.

Cache environment variables used by the scripts:
- Marker: `XDG_CACHE_HOME` and `MARKER_CACHE_DIR` -> `models/marker/`
- PaddleOCR: `PADDLEOCR_HOME`, `PADDLE_HOME`, and `HOME` -> `models/paddleocr/`
- Layout engine: `BOOKMIND_LAYOUT_MODEL_DIR` -> `models/layoutparser_publaynet/`
- Hugging Face: `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_CACHE` -> `models/qwen3_vl/*`

Qwen3-VL overrides:
- `BOOKMIND_QWEN3_VL_4B_MODEL_ID` selects the 4B model ID.
- `BOOKMIND_QWEN3_VL_8B_FP8_MODEL_ID` selects the 8B FP8 model ID.
- `BOOKMIND_QWEN3_VL_4B_DIR` and `BOOKMIND_QWEN3_VL_8B_FP8_DIR` set the snapshot paths.
- `BOOKMIND_QWEN3_VL_DIR` sets the local snapshot directory for serving.

For air-gapped vLLM usage, point vLLM to the local model path:
- Example: `python -m vllm.entrypoints.openai.api_server --model tools/bookmind_bench/offline_bundle/models/qwen3_vl/4b`

The bundle contents are intentionally git-ignored. Keep this folder alongside
this repo when transferring to an offline machine.
