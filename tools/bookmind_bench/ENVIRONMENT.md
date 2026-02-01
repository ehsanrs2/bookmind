# Bookmind Bench Offline Environment

This benchmark harness uses its own virtual environment so it never touches the
main Open WebUI runtime. The bench venv isolates heavier OCR/VLM dependencies
and lets you ship an offline wheel + model bundle to air-gapped machines.

## Overview
- Venv: `.venv_bookmind_bench/` (separate from any Open WebUI venv)
- Requirements: `tools/bookmind_bench/requirements/*.txt`
- Offline bundle: `tools/bookmind_bench/offline_bundle/` (git-ignored)

## Online prefetch (internet-connected machine)
1) Create the wheelhouse + download model assets:

```bash
./tools/bookmind_bench/scripts/prefetch_online.sh \
  --sample-pdf /path/to/sample.pdf \
  --sample-image /path/to/sample.png \
  --qwen-model Qwen/Qwen3-VL-4B-Instruct
```

Notes:
- `--sample-pdf` triggers a Marker warmup run to populate caches.
- `--sample-image` optionally triggers a PP-Structure warmup.
- Override the Qwen3-VL model with `--qwen-model` or `QWEN3_VL_MODEL=...`.

## Transfer to offline machine
1) Copy the entire repo *including* `tools/bookmind_bench/offline_bundle/`.
2) Verify that `offline_bundle/wheels/` and `offline_bundle/models/` are present.

## Offline install
```bash
./tools/bookmind_bench/scripts/install_offline.sh
```

This installs strictly from `offline_bundle/wheels` and sets cache environment
variables to the local bundle paths.

## Optional: create venv (online vs offline)
```bash
./tools/bookmind_bench/scripts/create_venv.sh --online
./tools/bookmind_bench/scripts/create_venv.sh --offline
```

## Offline verification
```bash
./tools/bookmind_bench/scripts/verify_offline.sh --pdf /path/to/sample.pdf
```

This enforces offline flags (HF/Transformers) and runs the CLI help commands.

## Common pitfalls
- **Dependency conflicts**: Keep bench dependencies isolated in
  `.venv_bookmind_bench/` to avoid Open WebUI runtime breakage.
- **Cache locations**: If models fail to load offline, confirm the env vars:
  - `XDG_CACHE_HOME` + `MARKER_CACHE_DIR` -> `offline_bundle/models/marker`
  - `PADDLEOCR_HOME` + `PADDLE_HOME` -> `offline_bundle/models/paddleocr`
  - `HF_HOME` + `HF_HUB_CACHE` + `TRANSFORMERS_CACHE` -> `offline_bundle/models/qwen3_vl`
- **GPU wheels**: `paddlepaddle-gpu` wheels are CUDA-specific; ensure the
  online machine downloads wheels compatible with the offline GPU stack.
- **File mode noise**: `core.filemode=false` is set locally to avoid noisy
  permission flips on filesystems that don't preserve POSIX modes.
