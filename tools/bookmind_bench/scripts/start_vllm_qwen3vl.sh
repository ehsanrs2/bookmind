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
LIMIT_MM="${BOOKMIND_VLLM_LIMIT_MM:-}"
COMPILE_MM_ENCODER="${BOOKMIND_VLLM_COMPILE_MM_ENCODER:-}"

usage() {
  cat <<'USAGE' >&2
Usage: start_vllm_qwen3vl.sh [--offline] [--model /path/or/hf-id] [--preset NAME] [--max-model-len N] [--gpu-mem 0.8] [--max-num-seqs N] [--max-num-batched-tokens N] [--limit-mm JSON] [--compile-mm-encoder true|false]
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
    --limit-mm)
      LIMIT_MM="$2"
      shift 2
      ;;
    --compile-mm-encoder)
      COMPILE_MM_ENCODER="$2"
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
  PRESET_LIMIT_MM=""
  PRESET_COMPILE_MM_ENCODER=""

  case "$PRESET" in
    4b_12gb_caption)
      PRESET_MAX_MODEL_LEN="1536"
      PRESET_GPU_MEM_UTIL="0.90"
      PRESET_MAX_NUM_SEQS="1"
      PRESET_MAX_NUM_BATCHED_TOKENS="1024"
      PRESET_LIMIT_MM='{"video":{"count":0},"image":{"count":1,"width":384,"height":384}}'
      PRESET_COMPILE_MM_ENCODER="false"
      ;;
    4b_12gb_safe)
      PRESET_MAX_MODEL_LEN="1536"
      PRESET_GPU_MEM_UTIL="0.90"
      PRESET_MAX_NUM_SEQS="1"
      PRESET_MAX_NUM_BATCHED_TOKENS="1024"
      PRESET_LIMIT_MM='{"video":{"count":0},"image":{"count":1,"width":384,"height":384}}'
      PRESET_COMPILE_MM_ENCODER="false"
      ;;
    8b_fp8_12gb_safe)
      PRESET_MAX_MODEL_LEN="4096"
      PRESET_GPU_MEM_UTIL="0.80"
      PRESET_MAX_NUM_SEQS="1"
      PRESET_MAX_NUM_BATCHED_TOKENS="1024"
      PRESET_LIMIT_MM='{"video":{"count":0},"image":{"count":1,"width":512,"height":512}}'
      PRESET_COMPILE_MM_ENCODER="false"
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
  if [[ -z "$LIMIT_MM" ]]; then
    LIMIT_MM="$PRESET_LIMIT_MM"
  fi
  if [[ -z "$COMPILE_MM_ENCODER" ]]; then
    COMPILE_MM_ENCODER="$PRESET_COMPILE_MM_ENCODER"
  fi
  if [[ "$PRESET" == "4b_12gb_caption" || "$PRESET" == "4b_12gb_safe" || "$PRESET" == "8b_fp8_12gb_safe" ]]; then
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
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
if [[ -n "$LIMIT_MM" ]]; then
  VLLM_ARGS+=("--limit-mm-per-prompt" "$LIMIT_MM")
fi
if [[ -n "$COMPILE_MM_ENCODER" ]]; then
  if [[ "$COMPILE_MM_ENCODER" != "true" && "$COMPILE_MM_ENCODER" != "false" ]]; then
    echo "Invalid --compile-mm-encoder value: $COMPILE_MM_ENCODER (use true|false)" >&2
    exit 1
  fi
  VLLM_ARGS+=("--compilation-config" "{\"compile_mm_encoder\":$COMPILE_MM_ENCODER}")
fi

echo "vLLM server started. Endpoint: $ENDPOINT_URL Model: $SERVED_MODEL_NAME ($MODEL)"
if [[ -n "${PYTORCH_ALLOC_CONF:-}" ]]; then
  echo "PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"
fi
if [[ -n "$LIMIT_MM" ]]; then
  echo "limit-mm-per-prompt=$LIMIT_MM"
fi
if [[ -n "$COMPILE_MM_ENCODER" ]]; then
  echo "compile-mm-encoder=$COMPILE_MM_ENCODER"
fi
exec vllm serve "$MODEL" "${VLLM_ARGS[@]}"
