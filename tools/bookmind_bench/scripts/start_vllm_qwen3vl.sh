#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv_bookmind_bench_vlm"
DEFAULT_MODEL_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle/models/qwen3_vl"
MODEL_DEFAULT="${BOOKMIND_QWEN3_VL_DIR:-$DEFAULT_MODEL_DIR}"

MODEL="$MODEL_DEFAULT"
OFFLINE=0
HOST="127.0.0.1"
PORT="8000"
SERVED_MODEL_NAME="${BOOKMIND_VLM_MODEL_NAME:-qwen3-vl}"

usage() {
  cat <<'USAGE' >&2
Usage: start_vllm_qwen3vl.sh [--offline] [--model /path/or/hf-id]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --offline)
      OFFLINE=1
      shift 1
      ;;
    --model)
      MODEL="$2"
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

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing VLM venv at $VENV_DIR." >&2
  echo "Run: ./tools/bookmind_bench/scripts/install_offline.sh --profile vlm" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ "$OFFLINE" -eq 1 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HOME="${BOOKMIND_HF_HOME:-$ROOT_DIR/tools/bookmind_bench/offline_bundle/models/_hf_home}"

  if [[ ! -d "$MODEL" && ! -f "$MODEL" ]]; then
    echo "Offline mode requires a local model path. Not found: $MODEL" >&2
    exit 1
  fi
fi

ENDPOINT_URL="http://$HOST:$PORT/v1"

echo "vLLM server started. Endpoint: $ENDPOINT_URL Model: $SERVED_MODEL_NAME ($MODEL)"
exec vllm serve "$MODEL" --host "$HOST" --port "$PORT" --served-model-name "$SERVED_MODEL_NAME" --trust-remote-code
