#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BENCH_PY="$ROOT_DIR/tools/bookmind_bench/scripts/bench_python.sh"
RUN_PY="$ROOT_DIR/tools/bookmind_bench/run.py"

usage() {
  cat <<USAGE
Usage:
  tools/bookmind_bench/scripts/e2e_bookmind.sh \
    --pdf <path> | --from_bundle <bundle_path> \
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
FROM_BUNDLE=""
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
    --from_bundle)
      FROM_BUNDLE="${2:-}"
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

if [[ -z "$OUT" ]]; then
  echo "--out is required." >&2
  usage
  exit 2
fi

if [[ -n "$PDF" && -n "$FROM_BUNDLE" ]]; then
  echo "Use exactly one of --pdf or --from_bundle." >&2
  usage
  exit 2
fi

if [[ -z "$PDF" && -z "$FROM_BUNDLE" ]]; then
  echo "Either --pdf or --from_bundle is required." >&2
  usage
  exit 2
fi

if [[ -n "$PDF" && ! -f "$PDF" ]]; then
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
BUNDLE_DIR="$OUT/bundle"

MODE="pdf"
if [[ -n "$FROM_BUNDLE" ]]; then
  MODE="from_bundle"
fi

if [[ "$MODE" == "from_bundle" ]]; then
  if [[ -f "$FROM_BUNDLE" ]]; then
    BUNDLE_DIR="$(cd "$(dirname "$FROM_BUNDLE")" && pwd)"
  elif [[ -d "$FROM_BUNDLE" ]]; then
    BUNDLE_DIR="$(cd "$FROM_BUNDLE" && pwd)"
  else
    echo "--from_bundle path not found: $FROM_BUNDLE" >&2
    exit 2
  fi
  if [[ ! -f "$BUNDLE_DIR/records.jsonl" ]]; then
    echo "Bundle records.jsonl not found in: $BUNDLE_DIR" >&2
    exit 2
  fi
fi

PAGE_COUNT="0"
PAGES="(none)"
MARKER_STATUS="SKIP"
RENDER_STATUS="SKIP"
PADDLEOCR_STATUS="SKIP"
LAYOUT_STATUS="SKIP"
VLM_LAYOUT_STATUS="SKIP"
MERGE_STATUS="SKIP"
BUNDLE_STATUS="SKIP"
QDRANT_INGEST_STATUS="SKIP"
RETRIEVAL_STATUS="SKIP"
EVAL_STATUS="SKIP"
RELIABILITY_STATUS="SKIP"

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

if [[ "$MODE" == "pdf" ]]; then
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
fi

if [[ "$MODE" == "pdf" ]]; then
  # (a) Extraction: marker + render/paddleocr; optionally layout + vlm-layout.
  if run_optional_step marker_single \
    "$BENCH_PY" "$RUN_PY" marker --pdf "$PDF" --out "$OUT"; then
    MARKER_STATUS="RUN"
  else
    MARKER_STATUS="SKIP_MISSING"
    echo "[E2E] WARN: marker_single unavailable, skipping marker extraction."
  fi

  run_step render \
    "$BENCH_PY" "$RUN_PY" render --pdf "$PDF" --out "$IMAGE_DIR" --pages "$PAGES"
  RENDER_STATUS="RUN"

  if run_optional_step paddleocr \
    "$BENCH_PY" "$RUN_PY" paddleocr --imgdir "$IMAGE_DIR" --out "$OUT"; then
    PADDLEOCR_STATUS="RUN"
  else
    PADDLEOCR_STATUS="SKIP_MISSING"
    echo "[E2E] ERROR: paddleocr unavailable and --pdf mode requires it." >&2
    exit 1
  fi

  LAYOUT_MODEL_DIR="${BOOKMIND_LAYOUT_MODEL_DIR:-$ROOT_DIR/tools/bookmind_bench/offline_bundle/models/layoutparser_publaynet}"
  if [[ -d "$LAYOUT_MODEL_DIR" ]] && find "$LAYOUT_MODEL_DIR" -type f | grep -q .; then
    if run_optional_step layout \
      "$BENCH_PY" "$RUN_PY" layout --imgdir "$IMAGE_DIR" --out "$OUT" --model_dir "$LAYOUT_MODEL_DIR"; then
      LAYOUT_STATUS="RUN"
    else
      LAYOUT_STATUS="SKIP_MISSING"
      echo "[E2E] WARN: layout extractor unavailable, skipping layout."
    fi

    if [[ "$BACKEND" == "ollama" ]]; then
      if run_optional_step vlm_layout \
        "$BENCH_PY" "$RUN_PY" vlm-layout \
        --imgdir "$IMAGE_DIR" \
        --out "$OUT" \
        --backend ollama \
        --ollama_url "$OLLAMA_URL" \
        --ollama_model "$OLLAMA_MODEL" \
        --ollama_api "$OLLAMA_API"; then
        VLM_LAYOUT_STATUS="RUN"
      else
        VLM_LAYOUT_STATUS="SKIP_MISSING"
        echo "[E2E] WARN: vlm-layout unavailable, skipping."
      fi
    fi
  else
    LAYOUT_STATUS="SKIP_MISSING"
    echo "[E2E] WARN: layout model dir not available, skipping layout/vlm-layout: $LAYOUT_MODEL_DIR"
  fi

  # (b) Merge + bundle.
  run_step merge \
    "$BENCH_PY" "$RUN_PY" merge --out "$OUT"
  MERGE_STATUS="RUN"

  run_step bundle \
    "$BENCH_PY" "$RUN_PY" bundle --out "$OUT" --imgdir "$IMAGE_DIR"
  BUNDLE_STATUS="RUN"
fi

# (c) Qdrant ingest.
run_step qdrant_ingest \
  "$BENCH_PY" "$RUN_PY" qdrant-ingest \
  --run "$OUT" \
  --bundle_dir "$BUNDLE_DIR" \
  --qdrant_url "$QDRANT_URL" \
  --collection "$COLLECTION"
QDRANT_INGEST_STATUS="RUN"

# (d) Optional evaluation suite.
if [[ -n "$QUERIES" ]]; then
  run_step retrieval_eval \
    "$BENCH_PY" "$RUN_PY" retrieval-eval \
    --queries "$QUERIES" \
    --out "$RETR_OUT" \
    --qdrant_url "$QDRANT_URL" \
    --collection "$COLLECTION"
  RETRIEVAL_STATUS="RUN"

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
  EVAL_STATUS="RUN"

  run_step reliability_pack \
    "$BENCH_PY" "$RUN_PY" reliability-pack \
    --retrieval_report "$RETR_OUT/retrieval_report.json" \
    --eval_report "$EVAL_OUT/report.json" \
    --out "$RELIABILITY_OUT"
  RELIABILITY_STATUS="RUN"
else
  echo "[E2E] --queries not provided; skipping retrieval-eval/eval/reliability-pack"
fi

# (e) Final summary.
{
  echo "# BookMind E2E Summary"
  echo
  echo "- mode: \\`$MODE\\`"
  echo "- pdf: \\`${PDF:-"(none)"}\\`"
  echo "- from_bundle: \\`${FROM_BUNDLE:-"(none)"}\\`"
  echo "- out: \\`$OUT\\`"
  echo "- bundle_dir_used: \\`$BUNDLE_DIR\\`"
  echo "- pages_processed: ${PAGE_COUNT:-0}"
  echo "- pages_range: \\`${PAGES:-"(none)"}\\`"
  echo "- qdrant_url: \\`$QDRANT_URL\\`"
  echo "- collection: \\`$COLLECTION\\`"
  echo "- backend: \\`${BACKEND:-default(eval)}\\`"
  echo "- ollama_url: \\`$OLLAMA_URL\\`"
  echo "- ollama_model: \\`$OLLAMA_MODEL\\`"
  echo "- ollama_api: \\`$OLLAMA_API\\`"
  if [[ -n "$QUERIES" ]]; then
    echo "- queries: \\`$QUERIES\\`"
  else
    echo "- queries: \\`(none)\\`"
  fi
  echo
  echo "## Step Status"
  echo
  echo "- marker_single: $MARKER_STATUS"
  echo "- render: $RENDER_STATUS"
  echo "- paddleocr: $PADDLEOCR_STATUS"
  echo "- layout: $LAYOUT_STATUS"
  echo "- vlm-layout: $VLM_LAYOUT_STATUS"
  echo "- merge: $MERGE_STATUS"
  echo "- bundle: $BUNDLE_STATUS"
  echo "- qdrant-ingest: $QDRANT_INGEST_STATUS"
  echo "- retrieval-eval: $RETRIEVAL_STATUS"
  echo "- eval: $EVAL_STATUS"
  echo "- reliability-pack: $RELIABILITY_STATUS"
  echo
  echo "## Key Outputs"
  echo
  echo "- rendered_images: \\`$IMAGE_DIR\\`"
  echo "- marker_output: \\`$OUT/marker/output.jsonl\\`"
  echo "- paddleocr_output: \\`$OUT/paddleocr/output.jsonl\\`"
  echo "- layout_output: \\`$OUT/layout/output.jsonl\\`"
  echo "- vlm_output: \\`$OUT/vlm/output.jsonl\\`"
  echo "- ingest_records: \\`$OUT/ingest/records.jsonl\\`"
  echo "- bundle_records: \\`$BUNDLE_DIR/records.jsonl\\`"
  echo "- retrieval_report: \\`$RETR_OUT/retrieval_report.json\\`"
  echo "- eval_report: \\`$EVAL_OUT/report.json\\`"
  echo "- reliability_report: \\`$RELIABILITY_OUT/reliability_report.json\\`"
  echo "- logs: \\`$OUT/logs\\`"
} > "$SUMMARY_PATH"

echo "[E2E] done. Summary: $SUMMARY_PATH"
