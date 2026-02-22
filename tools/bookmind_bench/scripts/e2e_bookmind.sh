#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BENCH_PY="$ROOT_DIR/tools/bookmind_bench/scripts/bench_python.sh"
RUN_PY="$ROOT_DIR/tools/bookmind_bench/run.py"

usage() {
  cat <<USAGE
Usage:
  tools/bookmind_bench/scripts/e2e_bookmind.sh \
    --pdf <path> \
    --out <dir> \
    [--qdrant_url <url>] \
    [--collection <name>] \
    [--backend ollama] \
    [--ollama_url <url>] \
    [--ollama_model <model>] \
    [--ollama_api <chat|generate>] \
    [--queries <truth_json>]

Notes:
- This script is a thin orchestrator over existing run.py subcommands.
- If --queries is omitted, retrieval/eval/reliability steps are skipped.
USAGE
}

PDF=""
OUT=""
QDRANT_URL="http://127.0.0.1:6333"
COLLECTION="bookmind_bench"
BACKEND=""
OLLAMA_URL="http://127.0.0.1:11434"
OLLAMA_MODEL="qwen3-vl:latest"
OLLAMA_API="generate"
QUERIES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdf)
      PDF="${2:-}"
      shift 2
      ;;
    --out)
      OUT="${2:-}"
      shift 2
      ;;
    --qdrant_url)
      QDRANT_URL="${2:-}"
      shift 2
      ;;
    --collection)
      COLLECTION="${2:-}"
      shift 2
      ;;
    --backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    --ollama_url)
      OLLAMA_URL="${2:-}"
      shift 2
      ;;
    --ollama_model)
      OLLAMA_MODEL="${2:-}"
      shift 2
      ;;
    --ollama_api)
      OLLAMA_API="${2:-}"
      shift 2
      ;;
    --queries)
      QUERIES="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$PDF" || -z "$OUT" ]]; then
  echo "--pdf and --out are required." >&2
  usage
  exit 2
fi

if [[ ! -f "$PDF" ]]; then
  echo "PDF not found: $PDF" >&2
  exit 2
fi

if [[ -n "$QUERIES" && ! -f "$QUERIES" ]]; then
  echo "Queries file not found: $QUERIES" >&2
  exit 2
fi

if [[ -n "$BACKEND" && "$BACKEND" != "ollama" ]]; then
  echo "Unsupported --backend '$BACKEND'. Only 'ollama' is accepted by this runner." >&2
  exit 2
fi

if [[ "$OLLAMA_API" != "chat" && "$OLLAMA_API" != "generate" ]]; then
  echo "Invalid --ollama_api '$OLLAMA_API'. Use chat or generate." >&2
  exit 2
fi

mkdir -p "$OUT" "$OUT/logs"

SUMMARY_PATH="$OUT/E2E_SUMMARY.md"
IMAGE_DIR="$OUT/images"
RETR_OUT="$OUT/retrieval_eval"
EVAL_OUT="$OUT/eval"
RELIABILITY_OUT="$OUT/reliability"

MARKER_RAN=0
LAYOUT_RAN=0
VLM_LAYOUT_RAN=0
RETRIEVAL_RAN=0
EVAL_RAN=0
RELIABILITY_RAN=0

run_step() {
  local step="$1"
  shift
  local log_path="$OUT/logs/${step}.log"
  echo "[E2E] step=${step}"
  "$@" 2>&1 | tee "$log_path"
}

run_optional_step() {
  local step="$1"
  shift
  local log_path="$OUT/logs/${step}.log"
  echo "[E2E] optional_step=${step}"
  if "$@" 2>&1 | tee "$log_path"; then
    return 0
  fi
  echo "[E2E] optional step failed and will be skipped: ${step}" | tee -a "$log_path"
  return 1
}

PAGE_COUNT="$(BOOKMIND_E2E_PDF="$PDF" "$BENCH_PY" - <<'PY'
import fitz
import os
pdf_path = os.environ['BOOKMIND_E2E_PDF']
with fitz.open(pdf_path) as doc:
    print(doc.page_count)
PY
)"

if [[ -z "$PAGE_COUNT" || "$PAGE_COUNT" -lt 1 ]]; then
  echo "Failed to determine page count for: $PDF" >&2
  exit 1
fi

PAGES="1-${PAGE_COUNT}"

# (a) Extraction: marker + render/paddleocr; optionally layout + vlm-layout.
run_step marker \
  "$BENCH_PY" "$RUN_PY" marker --pdf "$PDF" --out "$OUT"
MARKER_RAN=1

run_step render \
  "$BENCH_PY" "$RUN_PY" render --pdf "$PDF" --out "$IMAGE_DIR" --pages "$PAGES"

run_step paddleocr \
  "$BENCH_PY" "$RUN_PY" paddleocr --imgdir "$IMAGE_DIR" --out "$OUT"

LAYOUT_MODEL_DIR="${BOOKMIND_LAYOUT_MODEL_DIR:-$ROOT_DIR/tools/bookmind_bench/offline_bundle/models/layoutparser_publaynet}"
if [[ -d "$LAYOUT_MODEL_DIR" ]] && find "$LAYOUT_MODEL_DIR" -type f | grep -q .; then
  run_step layout \
    "$BENCH_PY" "$RUN_PY" layout --imgdir "$IMAGE_DIR" --out "$OUT" --model_dir "$LAYOUT_MODEL_DIR"
  LAYOUT_RAN=1

  if [[ "$BACKEND" == "ollama" ]]; then
    if run_optional_step vlm_layout \
      "$BENCH_PY" "$RUN_PY" vlm-layout \
      --imgdir "$IMAGE_DIR" \
      --out "$OUT" \
      --backend ollama \
      --ollama_url "$OLLAMA_URL" \
      --ollama_model "$OLLAMA_MODEL" \
      --ollama_api "$OLLAMA_API"; then
      VLM_LAYOUT_RAN=1
    fi
  fi
else
  echo "[E2E] layout model dir not available, skipping layout/vlm-layout: $LAYOUT_MODEL_DIR"
fi

# (b) Merge + bundle.
run_step merge \
  "$BENCH_PY" "$RUN_PY" merge --out "$OUT"

run_step bundle \
  "$BENCH_PY" "$RUN_PY" bundle --out "$OUT" --imgdir "$IMAGE_DIR"

# (c) Qdrant ingest.
run_step qdrant_ingest \
  "$BENCH_PY" "$RUN_PY" qdrant-ingest \
  --run "$OUT" \
  --qdrant_url "$QDRANT_URL" \
  --collection "$COLLECTION"

# (d) Optional evaluation suite.
if [[ -n "$QUERIES" ]]; then
  run_step retrieval_eval \
    "$BENCH_PY" "$RUN_PY" retrieval-eval \
    --queries "$QUERIES" \
    --out "$RETR_OUT" \
    --qdrant_url "$QDRANT_URL" \
    --collection "$COLLECTION"
  RETRIEVAL_RAN=1

  if [[ "$BACKEND" == "ollama" ]]; then
    run_step eval \
      "$BENCH_PY" "$RUN_PY" eval \
      --queries "$QUERIES" \
      --out "$EVAL_OUT" \
      --qdrant_url "$QDRANT_URL" \
      --collection "$COLLECTION" \
      --backend ollama \
      --ollama_url "$OLLAMA_URL" \
      --ollama_model "$OLLAMA_MODEL" \
      --ollama_api "$OLLAMA_API"
  else
    run_step eval \
      "$BENCH_PY" "$RUN_PY" eval \
      --queries "$QUERIES" \
      --out "$EVAL_OUT" \
      --qdrant_url "$QDRANT_URL" \
      --collection "$COLLECTION"
  fi
  EVAL_RAN=1

  run_step reliability_pack \
    "$BENCH_PY" "$RUN_PY" reliability-pack \
    --retrieval_report "$RETR_OUT/retrieval_report.json" \
    --eval_report "$EVAL_OUT/report.json" \
    --out "$RELIABILITY_OUT"
  RELIABILITY_RAN=1
else
  echo "[E2E] --queries not provided; skipping retrieval-eval/eval/reliability-pack"
fi

# (e) Final summary.
{
  echo "# BookMind E2E Summary"
  echo
  echo "- pdf: \\`$PDF\\`"
  echo "- out: \\`$OUT\\`"
  echo "- pages_processed: $PAGE_COUNT"
  echo "- qdrant_url: \\`$QDRANT_URL\\`"
  echo "- collection: \\`$COLLECTION\\`"
  echo "- backend: \\`${BACKEND:-default(eval)}\\`"
  if [[ -n "$QUERIES" ]]; then
    echo "- queries: \\`$QUERIES\\`"
  else
    echo "- queries: (not provided)"
  fi
  echo
  echo "## Step Status"
  echo
  echo "- marker: $MARKER_RAN"
  echo "- layout: $LAYOUT_RAN"
  echo "- vlm-layout: $VLM_LAYOUT_RAN"
  echo "- retrieval-eval: $RETRIEVAL_RAN"
  echo "- eval: $EVAL_RAN"
  echo "- reliability-pack: $RELIABILITY_RAN"
  echo
  echo "## Key Outputs"
  echo
  echo "- rendered_images: \\`$IMAGE_DIR\\`"
  echo "- marker_output: \\`$OUT/marker/output.jsonl\\`"
  echo "- paddleocr_output: \\`$OUT/paddleocr/output.jsonl\\`"
  echo "- layout_output: \\`$OUT/layout/output.jsonl\\`"
  echo "- vlm_output: \\`$OUT/vlm/output.jsonl\\`"
  echo "- ingest_records: \\`$OUT/ingest/records.jsonl\\`"
  echo "- bundle_records: \\`$OUT/bundle/records.jsonl\\`"
  echo "- retrieval_report: \\`$RETR_OUT/retrieval_report.json\\`"
  echo "- eval_report: \\`$EVAL_OUT/report.json\\`"
  echo "- reliability_report: \\`$RELIABILITY_OUT/reliability_report.json\\`"
  echo "- logs: \\`$OUT/logs\\`"
} > "$SUMMARY_PATH"

echo "[E2E] done. Summary: $SUMMARY_PATH"
