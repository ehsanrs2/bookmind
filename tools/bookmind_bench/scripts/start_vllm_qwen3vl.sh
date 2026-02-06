#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv_bookmind_bench_vlm"
DEFAULT_MODEL_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle/models/qwen3_vl/4b"
MODEL_DEFAULT="${BOOKMIND_QWEN3_VL_DIR:-$DEFAULT_MODEL_DIR}"

MODEL="$MODEL_DEFAULT"
OFFLINE=0
HOST="127.0.0.1"
PORT="8000"
SERVED_MODEL_NAME="${BOOKMIND_VLM_MODEL_NAME:-qwen3-vl}"
PRESET="${BOOKMIND_VLLM_PRESET-4b_12gb_safe}"
MAX_MODEL_LEN="${BOOKMIND_VLLM_MAX_MODEL_LEN:-}"
GPU_MEM_UTIL="${BOOKMIND_VLLM_GPU_MEMORY_UTILIZATION:-}"
MAX_NUM_SEQS="${BOOKMIND_VLLM_MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${BOOKMIND_VLLM_MAX_NUM_BATCHED_TOKENS:-}"

usage() {
  cat <<'USAGE' >&2
Usage: start_vllm_qwen3vl.sh [--offline] [--model /path/or/hf-id] [--preset NAME] [--max-model-len N] [--gpu-mem 0.8] [--max-num-seqs N] [--max-num-batched-tokens N]
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
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --gpu-mem)
      GPU_MEM_UTIL="$2"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
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

if [[ -n "$PRESET" ]]; then
  PRESET_MAX_MODEL_LEN=""
  PRESET_GPU_MEM_UTIL=""
  PRESET_MAX_NUM_SEQS=""
  PRESET_MAX_NUM_BATCHED_TOKENS=""

  case "$PRESET" in
    4b_12gb_safe)
      PRESET_MAX_MODEL_LEN="4096"
      PRESET_GPU_MEM_UTIL="0.80"
      PRESET_MAX_NUM_SEQS="1"
      PRESET_MAX_NUM_BATCHED_TOKENS="2048"
      ;;
    8b_fp8_12gb_safe)
      PRESET_MAX_MODEL_LEN="4096"
      PRESET_GPU_MEM_UTIL="0.80"
      PRESET_MAX_NUM_SEQS="1"
      PRESET_MAX_NUM_BATCHED_TOKENS="1024"
      ;;
    high_mem_default)
      PRESET_MAX_MODEL_LEN="8192"
      PRESET_GPU_MEM_UTIL="0.90"
      PRESET_MAX_NUM_SEQS="2"
      PRESET_MAX_NUM_BATCHED_TOKENS="4096"
      ;;
    *)
      echo "Unknown preset: $PRESET" >&2
      usage
      exit 1
      ;;
  esac

  if [[ -z "$MAX_MODEL_LEN" ]]; then
    MAX_MODEL_LEN="$PRESET_MAX_MODEL_LEN"
  fi
  if [[ -z "$GPU_MEM_UTIL" ]]; then
    GPU_MEM_UTIL="$PRESET_GPU_MEM_UTIL"
  fi
  if [[ -z "$MAX_NUM_SEQS" ]]; then
    MAX_NUM_SEQS="$PRESET_MAX_NUM_SEQS"
  fi
  if [[ -z "$MAX_NUM_BATCHED_TOKENS" ]]; then
    MAX_NUM_BATCHED_TOKENS="$PRESET_MAX_NUM_BATCHED_TOKENS"
  fi
fi

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

VLLM_ARGS=("--host" "$HOST" "--port" "$PORT" "--served-model-name" "$SERVED_MODEL_NAME" "--trust-remote-code")
if [[ -n "$MAX_MODEL_LEN" ]]; then
  VLLM_ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
fi
if [[ -n "$GPU_MEM_UTIL" ]]; then
  VLLM_ARGS+=("--gpu-memory-utilization" "$GPU_MEM_UTIL")
fi
if [[ -n "$MAX_NUM_SEQS" ]]; then
  VLLM_ARGS+=("--max-num-seqs" "$MAX_NUM_SEQS")
fi
if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
  VLLM_ARGS+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")
fi

echo "vLLM server started. Endpoint: $ENDPOINT_URL Model: $SERVED_MODEL_NAME ($MODEL)"
exec vllm serve "$MODEL" "${VLLM_ARGS[@]}"
