#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

RUN_DIR=""
ENDPOINT="http://127.0.0.1:8000/v1"
MODEL="qwen3-vl"
MAX_TOKENS="256"
TEMPERATURE="0.2"

usage() {
  cat <<'EOF' >&2
Usage: run_vlm_captions.sh --run <RUN_DIR> [--endpoint <URL>] [--model <NAME>] [--max-tokens <int>] [--temperature <float>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_DIR="$2"
      shift 2
      ;;
    --endpoint)
      ENDPOINT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "Missing required --run <RUN_DIR>." >&2
  usage
  exit 1
fi

JOBS_PATH="$RUN_DIR/paddleocr/vlm_jobs.jsonl"
if [[ ! -f "$JOBS_PATH" ]]; then
  echo "Missing VLM jobs JSONL: $JOBS_PATH" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/vlm"

python "$ROOT_DIR/tools/bookmind_bench/run.py" vlm \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --jobs "$JOBS_PATH" \
  --out "$RUN_DIR" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE"

OUTPUT_PATH="$RUN_DIR/vlm/output.jsonl"
echo "VLM output: $OUTPUT_PATH"
if [[ -f "$OUTPUT_PATH" ]]; then
  echo "VLM output preview:"
  head -n 2 "$OUTPUT_PATH" || true
else
  echo "No VLM output found at $OUTPUT_PATH"
fi
