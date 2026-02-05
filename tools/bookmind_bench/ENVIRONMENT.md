# Bookmind Bench Offline Environment

This benchmark harness uses its own virtual environments so it never touches
the main Open WebUI runtime. Bench venvs isolate heavier OCR/VLM dependencies
and let you ship an offline wheel + model bundle to air-gapped machines.

## Overview
- Venvs:
  - `.venv_bookmind_bench_marker/`
  - `.venv_bookmind_bench_ocr/`
  - `.venv_bookmind_bench_vlm/`
- Requirements: `tools/bookmind_bench/requirements/*.txt`
- Offline bundle: `tools/bookmind_bench/offline_bundle/` (git-ignored)
- PaddleOCR/PP-Structure must be installed in the bench venvs (not the app venv)

Profiles keep dependency stacks isolated:
- `marker`: base + marker requirements
- `ocr`: base + PaddleOCR requirements
- `vlm`: base + VLM requirements
- `dev`: optional dev tooling (`--dev`) installed into any chosen profile env

Legacy note: `.venv_bookmind_bench/` is still accepted by scripts as a fallback,
but new installs should use the per-profile venvs above.

## Online prefetch (internet-connected machine)
1) Create the wheelhouse + download model assets:

```bash
./tools/bookmind_bench/scripts/prefetch_online.sh --profile all \
  --sample-pdf /path/to/sample.pdf \
  --sample-image /path/to/sample.png
```

Notes:
- `--sample-pdf` triggers a Marker warmup run to populate caches.
- `--sample-image` optionally triggers a PP-Structure warmup.
- Override the Qwen3-VL model with `--qwen-model` or `BOOKMIND_QWEN3_VL_MODEL_ID=...`.
- Override the snapshot directory with `BOOKMIND_QWEN3_VL_DIR=...`.

## Transfer to offline machine
1) Copy the entire repo *including* `tools/bookmind_bench/offline_bundle/`.
2) Verify that `offline_bundle/wheels/` and `offline_bundle/models/` are present.

## Offline install
```bash
./tools/bookmind_bench/scripts/install_offline.sh --profile all
```

This installs strictly from `offline_bundle/wheels` and sets cache environment
variables to the local bundle paths.

## Optional: create venv (online vs offline)
```bash
./tools/bookmind_bench/scripts/create_venv.sh --online --profile marker
./tools/bookmind_bench/scripts/create_venv.sh --offline --profile ocr
./tools/bookmind_bench/scripts/create_venv.sh --online --profile vlm --dev
```

## Offline verification
```bash
./tools/bookmind_bench/scripts/verify_offline.sh --profile all \
  --pdf /path/to/sample.pdf \
  --image /path/to/sample.png
```

This enforces offline flags (HF/Transformers) and runs the CLI help commands.

## Qwen3-VL (vLLM) workflow
Online prefetch into the offline bundle:
```bash
BOOKMIND_QWEN3_VL_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct \
  ./tools/bookmind_bench/scripts/prefetch_online.sh --profile vlm
```

Start the server in offline mode (OpenAI-compatible):
```bash
./tools/bookmind_bench/scripts/start_vllm_qwen3vl.sh --offline \
  --model tools/bookmind_bench/offline_bundle/models/qwen3_vl
```

Run the smoke test:
```bash
python tools/bookmind_bench/scripts/vlm_smoke_test.py \
  --image /path/to/crop.png
```

## Common pitfalls
- **Dependency conflicts**: Keep bench dependencies isolated in the per-profile
  venvs to avoid Open WebUI runtime breakage and numpy/opencv/torch conflicts.
- **Cache locations**: If models fail to load offline, confirm the env vars:
  - `XDG_CACHE_HOME` + `MARKER_CACHE_DIR` -> `offline_bundle/models/marker`
  - `PADDLEOCR_HOME` + `PADDLE_HOME` + `HOME` -> `offline_bundle/models/paddleocr`
  - `HF_HOME` + `HF_HUB_CACHE` + `TRANSFORMERS_CACHE` -> `offline_bundle/models/qwen3_vl`
- **GPU wheels**: `paddlepaddle-gpu` wheels are CUDA-specific; ensure the
  online machine downloads wheels compatible with the offline GPU stack.
- **File mode noise**: `core.filemode=false` is set locally to avoid noisy
  permission flips on filesystems that don't preserve POSIX modes.
