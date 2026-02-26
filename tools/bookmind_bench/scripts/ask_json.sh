#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BENCH_PY="$ROOT_DIR/tools/bookmind_bench/scripts/bench_python.sh"
RUN_PY="$ROOT_DIR/tools/bookmind_bench/run.py"

QUERY=""
QDRANT_URL="http://127.0.0.1:6333"
COLLECTION="bookmind_bench"
BACKEND="ollama"
OLLAMA_URL="http://127.0.0.1:11434"
OLLAMA_MODEL="qwen3-vl:latest"
OLLAMA_API="generate"
FORCE_CITATIONS=""
ENFORCE_VERIFIED=""
OUT=""

usage() {
  cat <<'USAGE' >&2
Usage:
  tools/bookmind_bench/scripts/ask_json.sh \
    --query <text> \
    [--qdrant_url <url>] \
    [--collection <name>] \
    [--backend <ollama|vllm>] \
    [--ollama_url <url>] \
    [--ollama_model <model>] \
    [--ollama_api <chat|generate>] \
    [--force_citations <true|false>] \
    [--enforce_verified <true|false>] \
    [--out <path>]

Notes:
- Thin wrapper around: run.py ask
- Always prints final ask JSON to stdout.
- If --out is omitted, a temp file is used and then printed to stdout.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --query)
      QUERY="${2:-}"
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
    --force_citations)
      FORCE_CITATIONS="${2:-}"
      shift 2
      ;;
    --enforce_verified)
      ENFORCE_VERIFIED="${2:-}"
      shift 2
      ;;
    --out)
      OUT="${2:-}"
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

if [[ -z "$QUERY" ]]; then
  echo "--query is required." >&2
  usage
  exit 2
fi

if [[ "$BACKEND" != "ollama" && "$BACKEND" != "vllm" ]]; then
  echo "Invalid --backend '$BACKEND'. Use ollama or vllm." >&2
  exit 2
fi

if [[ "$OLLAMA_API" != "chat" && "$OLLAMA_API" != "generate" ]]; then
  echo "Invalid --ollama_api '$OLLAMA_API'. Use chat or generate." >&2
  exit 2
fi

ASK_ARGS=(
  ask
  --query "$QUERY"
  --qdrant_url "$QDRANT_URL"
  --collection "$COLLECTION"
  --backend "$BACKEND"
  --ollama_url "$OLLAMA_URL"
  --ollama_model "$OLLAMA_MODEL"
  --ollama_api "$OLLAMA_API"
)

if [[ -n "$FORCE_CITATIONS" ]]; then
  ASK_ARGS+=(--force_citations "$FORCE_CITATIONS")
fi

if [[ -n "$ENFORCE_VERIFIED" ]]; then
  ASK_ARGS+=(--enforce_verified "$ENFORCE_VERIFIED")
fi

if [[ -n "$OUT" ]]; then
  ASK_ARGS+=(--out "$OUT")
  "$BENCH_PY" "$RUN_PY" "${ASK_ARGS[@]}"
  exit 0
fi

TMP_OUT="$(mktemp -t bookmind_ask_json.XXXXXX.json)"
trap 'rm -f "$TMP_OUT"' EXIT

ASK_ARGS+=(--out "$TMP_OUT")
"$BENCH_PY" "$RUN_PY" "${ASK_ARGS[@]}" > /dev/null
cat "$TMP_OUT"
