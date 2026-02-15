# Bookmind Bench Offline Environment

This benchmark harness uses its own virtual environments so it never touches
the main Open WebUI runtime. Bench venvs isolate heavier OCR/VLM dependencies
and let you ship an offline wheel + model bundle to air-gapped machines.

## Overview
- Venvs:
  - `.venv_bookmind_bench_marker/`
  - `.venv_bookmind_bench_ocr/`
  - `.venv_bookmind_bench_layout/`
  - `.venv_bookmind_bench_vlm/`
  - `.venv_bookmind_bench_qdrant/`
- Requirements: `tools/bookmind_bench/requirements/*.txt`
- Offline bundle: `tools/bookmind_bench/offline_bundle/` (git-ignored)
- PaddleOCR/PP-Structure must be installed in the bench venvs (not the app venv)

Profiles keep dependency stacks isolated:
- `marker`: base + marker requirements
- `ocr`: base + PaddleOCR requirements
- `layout`: base + LayoutParser + PaddleOCR requirements + local Detectron2 wheel
- `vlm`: base + VLM requirements
- `qdrant`: base + Qdrant client + sentence-transformers requirements
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
- `--profile layout` also attempts to prewarm LayoutParser PubLayNet assets into
  `offline_bundle/models/layoutparser_publaynet`.
- Override Qwen3-VL download IDs with `BOOKMIND_QWEN3_VL_4B_MODEL_ID=...` or
  `BOOKMIND_QWEN3_VL_8B_FP8_MODEL_ID=...`.
- Override snapshot directories with `BOOKMIND_QWEN3_VL_4B_DIR=...` or
  `BOOKMIND_QWEN3_VL_8B_FP8_DIR=...` (or set `BOOKMIND_QWEN3_VL_DIR=...` for serving).

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
./tools/bookmind_bench/scripts/create_venv.sh --offline --profile layout
./tools/bookmind_bench/scripts/create_venv.sh --online --profile vlm --dev
./tools/bookmind_bench/scripts/create_venv.sh --offline --profile qdrant
```

## Qdrant ingestion (offline)
Embedding model default: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, small SBERT for technical English).

Optional local Qdrant service:
```bash
docker run --rm -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Prepare offline assets for qdrant profile (online machine):
```bash
./tools/bookmind_bench/scripts/prefetch_online.sh --profile qdrant
```

This stores:
- wheels in `tools/bookmind_bench/offline_bundle/wheels/qdrant/`
- embedding snapshot in `tools/bookmind_bench/offline_bundle/models/embeddings/all-MiniLM-L6-v2/`

Ingest a run bundle into Qdrant:
```bash
python tools/bookmind_bench/run.py bundle --out <RUN_DIR>
python tools/bookmind_bench/run.py qdrant-ingest --run <RUN_DIR> \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --id_mode uint64
```

Search:
```bash
python tools/bookmind_bench/run.py qdrant-search \
  --query "phase diagram calibration conditions" \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --top_k 10
```

Optional type filter:
```bash
python tools/bookmind_bench/run.py qdrant-search --query "..." --content_types text,table
```

## RAG preview (offline)
1) Build bundle and ingest into Qdrant:
```bash
python tools/bookmind_bench/run.py bundle --out <RUN_DIR>
python tools/bookmind_bench/run.py qdrant-ingest --run <RUN_DIR> \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --id_mode uint64
```

2) Start Qdrant (optional local Docker):
```bash
docker run --rm -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

3) Run RAG preview with Ollama backend (default):
```bash
python tools/bookmind_bench/run.py rag-preview \
  --query "Summarize the calibration setup" \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --backend ollama \
  --ollama_url http://127.0.0.1:11434 \
  --ollama_model qwen3-vl:latest
```

4) Run RAG preview with vLLM OpenAI-compatible backend:
```bash
python tools/bookmind_bench/run.py rag-preview \
  --query "Summarize the calibration setup" \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --backend vllm \
  --endpoint http://127.0.0.1:8000/v1 \
  --model qwen3-vl
```

Notes:
- Answers are instructed to cite as `[page:stable_id]`.
- `qdrant-ingest` uses Qdrant-compatible point IDs (`--id_mode uint64|uuid`, default `uint64`).
- Original `stable_id` is always preserved in payload, so citations remain stable_id-based.
- Use `--content_types text,table,figure_caption` to filter retrieval scope.
- Use `--max_context_chars` to bound prompt context size.
- Use `--show_snippets true` to print retrieved snippet previews in the output.

## Evaluation pack (offline, weakly supervised)
Use the evaluation harness to run repeatable multi-query RAG checks and compare
configurations (OCR/layout pipelines, retrieval limits, and backend variants).

Input query set:
- built-in sample: `tools/bookmind_bench/samples/queries_scanned_tech.json`
- schema per query:
  - `{"id":"q1","query":"...","notes":"optional","expected_pages":[2,3]}`
  - `expected_pages` is optional; when present it enables `hit@k_pages`.

Example after ingesting a run (for example `/tmp/bookmind_layout_run`):
```bash
python tools/bookmind_bench/run.py eval \
  --queries tools/bookmind_bench/samples/queries_scanned_tech.json \
  --out /tmp/bookmind_layout_run/eval_ollama \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --backend ollama \
  --ollama_url http://127.0.0.1:11434 \
  --ollama_model qwen3-vl:latest \
  --top_k 8 \
  --max_context_chars 6000 \
  --content_types text,table,figure_caption
```

vLLM variant:
```bash
python tools/bookmind_bench/run.py eval \
  --queries tools/bookmind_bench/samples/queries_scanned_tech.json \
  --out /tmp/bookmind_layout_run/eval_vllm \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --backend vllm \
  --endpoint http://127.0.0.1:8000/v1 \
  --model qwen3-vl \
  --top_k 12 \
  --max_context_chars 8000
```

Output artifacts:
- `eval/report.json`: full run metadata + per-query outputs + metrics
- `eval/report.md`: concise summary with per-query bullets, slowest queries,
  and lowest-citation queries
- `eval/items/<query-id>.json`: per-query artifact for diffs/regressions
- `eval/per_query/qNN_raw_provider.json`: provider issue details on failures
  (`generation_exception` / `empty_answer`)
- `eval/per_query/qNN_ollama_request.json` + `qNN_ollama_response.json`:
  first Ollama attempt request/response debug (on empty/failure)
- `eval/per_query/qNN_attempt1_*.json`, `qNN_attempt2_*.json`:
  retry request/response debug snapshots (on empty/failure)

Ollama empty-answer retry behavior:
- Initial request uses configured generation settings.
- Retry 1 reduces retrieved context in the prompt to a shorter context window.
- Retry 2 also simplifies the system instruction and uses safer generation
  settings (`temperature=0.0`, `max_tokens<=256`).
- If all attempts still return empty/whitespace (or provider errors), the query
  is marked `status=failed` with `failure_reason` and artifacts are preserved.

Heuristic note:
- `--heuristic_boost_figures true` (default) moves `figure_caption` retrievals
  earlier when query text suggests diagrams (`figure`, `diagram`, `schematic`,
  `block diagram`, `circuit`, `wiring`).

Why weakly supervised:
- No gold answers are required.
- Metrics focus on health signals: latency, citation presence/rate, retrieved ids,
  answer length, and optional page-hit checks via `expected_pages`.

Offline flags and model lookup:
- set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
- default local embedding path:
  `tools/bookmind_bench/offline_bundle/models/embeddings/all-MiniLM-L6-v2`
- override with `--embed_model /path/to/local/model` when needed.

## Offline verification
```bash
./tools/bookmind_bench/scripts/verify_offline.sh --profile all \
  --pdf /path/to/sample.pdf \
  --image /path/to/sample.png
```

This enforces offline flags (HF/Transformers) and runs the CLI help commands.
For `--profile layout`, a one-page layout smoke run is performed when `--image`
is provided.

## Layout profile (Detectron2 from source)
- Layout venv must use Python `3.10` (preferred) or `3.11`.
- `prefetch_online.sh --profile layout` now:
  - creates `.venv_bookmind_bench_layout` with Python 3.10/3.11
  - installs layout requirements (without Detectron2 from PyPI)
  - builds a pinned Detectron2 wheel from source
  - stores wheels under `tools/bookmind_bench/offline_bundle/wheels/layout/`
- `install_offline.sh --profile layout` installs strictly from
  `offline_bundle/wheels/layout/` (including Detectron2).

System prerequisites for layout (online build machine):
- Python `3.10` or `3.11`, plus `venv` module and matching dev headers (`python3.10-dev` or `python3.11-dev`)
- `build-essential`, `cmake`, `ninja-build`, `git`

Example:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git python3.10 python3.10-venv python3.10-dev
```

Online prefetch (layout):
```bash
./tools/bookmind_bench/scripts/prefetch_online.sh --profile layout --sample-image /path/to/sample.png
```

Offline install (layout):
```bash
./tools/bookmind_bench/scripts/install_offline.sh --profile layout
```

Offline verify (layout):
```bash
./tools/bookmind_bench/scripts/verify_offline.sh --profile layout --image /path/to/sample.png
```

## Qwen3-VL (vLLM) workflow
Online prefetch into the offline bundle (default 4B snapshot):
```bash
./tools/bookmind_bench/scripts/prefetch_online.sh --profile vlm --qwen3vl 4b
```

**12GB VRAM (caption)**
Start the server in offline mode (OpenAI-compatible) on a 12GB GPU:
```bash
./tools/bookmind_bench/scripts/start_vllm_qwen3vl.sh --offline --preset 4b_12gb_caption
```

Why this preset matters: vLLM + Qwen3-VL may OOM during initialization because
the engine profiles a maximum-size video input. The 12GB presets disable video,
cap image size, and lower `max_model_len` to fit KV cache requirements. For
captioning, ~1.5k tokens of context is typically enough. The presets also
disable multimodal encoder compilation to avoid extra overhead on low VRAM.
If you want to re-enable encoder compilation, pass `--compile-mm-encoder true`.

By default the 12GB presets apply `--limit-mm-per-prompt` (video count `0`,
image `384x384`). You can override the image limit if needed:

```bash
./tools/bookmind_bench/scripts/start_vllm_qwen3vl.sh --offline --preset 4b_12gb_caption \
  --limit-mm '{"video":{"count":0},"image":{"count":1,"width":384,"height":384}}'
```

Run the smoke test:
```bash
python tools/bookmind_bench/scripts/vlm_smoke_test.py \
  --image /path/to/crop.png
```

## Quickstart (VLM captions)
1) Start the server with the 12GB preset:
```bash
./tools/bookmind_bench/scripts/start_vllm_qwen3vl.sh --offline --preset 4b_12gb_caption
```

The 12GB preset keeps multimodal encoder compilation off (`--compile-mm-encoder false`) to reduce VRAM pressure.

2) Run the smoke test:
```bash
python tools/bookmind_bench/scripts/vlm_smoke_test.py \
  --image /path/to/crop.png
```

3) Run the VLM bench:
```bash
python tools/bookmind_bench/run.py vlm \
  --endpoint http://127.0.0.1:8000/v1 \
  --model qwen3-vl \
  --jobs <path>/vlm_jobs.jsonl \
  --out <outdir>
```

Quick regression check: the server should start without CUDA OOM during
initialization (before any requests are sent).

## Ollama backend (12GB VRAM)
Ollama can be used as the VLM caption backend without changing output schema.
Expected local model tag: `qwen3-vl:latest` (must already exist in Ollama).

Sanity-check Ollama and model tag (offline-safe, no pulls):
```bash
./tools/bookmind_bench/scripts/start_ollama_qwen3vl.sh
```

Run `vlm-layout` with Ollama:
```bash
python tools/bookmind_bench/run.py vlm-layout \
  --backend ollama \
  --ollama_url http://127.0.0.1:11434 \
  --ollama_model qwen3-vl:latest \
  --imgdir <IMG_DIR> \
  --out <RUN_DIR>
```

## Generate VLM captions for a run directory
1) Start the server:
```bash
./tools/bookmind_bench/scripts/start_vllm_qwen3vl.sh --offline --preset 4b_12gb_caption
```

2) Run captions from the PaddleOCR run directory:
```bash
./tools/bookmind_bench/scripts/run_vlm_captions.sh --run <RUN_DIR>
```

3) Merge VLM captions with OCR output:
```bash
python tools/bookmind_bench/run.py merge --out <RUN_DIR>
```

**More VRAM**
On larger GPUs, use the high-memory preset and a larger model snapshot. You can
also increase `max_model_len` and image size for longer contexts or higher-res
inputs. If memory headroom allows, test `--compile-mm-encoder true` for possible
throughput gains.

```bash
./tools/bookmind_bench/scripts/prefetch_online.sh --profile vlm --qwen3vl 8b_fp8
./tools/bookmind_bench/scripts/start_vllm_qwen3vl.sh --offline \
  --model tools/bookmind_bench/offline_bundle/models/qwen3_vl/8b_fp8 \
  --preset high_mem_default
```

## Common pitfalls
- **Dependency conflicts**: Keep bench dependencies isolated in the per-profile
  venvs to avoid Open WebUI runtime breakage and numpy/opencv/torch conflicts.
- **Cache locations**: If models fail to load offline, confirm the env vars:
  - `XDG_CACHE_HOME` + `MARKER_CACHE_DIR` -> `offline_bundle/models/marker`
  - `PADDLEOCR_HOME` + `PADDLE_HOME` + `HOME` -> `offline_bundle/models/paddleocr`
  - `BOOKMIND_LAYOUT_MODEL_DIR` -> `offline_bundle/models/layoutparser_publaynet`
  - `HF_HOME` + `HF_HUB_CACHE` + `TRANSFORMERS_CACHE` -> `offline_bundle/models/qwen3_vl/*`
- **GPU wheels**: `paddlepaddle-gpu` wheels are CUDA-specific; ensure the
  online machine downloads wheels compatible with the offline GPU stack.
- **File mode noise**: `core.filemode=false` is set locally to avoid noisy
  permission flips on filesystems that don't preserve POSIX modes.
