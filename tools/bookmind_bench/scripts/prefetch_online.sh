#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REQ_DIR="$ROOT_DIR/tools/bookmind_bench/requirements"
BUNDLE_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle"
WHEEL_DIR="$BUNDLE_DIR/wheels"
MODEL_DIR="$BUNDLE_DIR/models"
MARKER_DIR="$MODEL_DIR/marker"
PADDLE_DIR="$MODEL_DIR/paddleocr"
QWEN_DIR_BASE="$MODEL_DIR/qwen3_vl"
QWEN_4B_DIR_DEFAULT="$QWEN_DIR_BASE/4b"
QWEN_8B_FP8_DIR_DEFAULT="$QWEN_DIR_BASE/8b_fp8"

SAMPLE_PDF=""
SAMPLE_IMAGE=""
PROFILE="all"
QWEN3VL="4b"
MODEL_ID_4B_DEFAULT="Qwen/Qwen3-VL-4B-Instruct"
MODEL_ID_8B_FP8_DEFAULT="Qwen/Qwen3-VL-8B-Instruct-FP8"
MODEL_ID_4B="${BOOKMIND_QWEN3_VL_4B_MODEL_ID:-$MODEL_ID_4B_DEFAULT}"
MODEL_ID_8B_FP8="${BOOKMIND_QWEN3_VL_8B_FP8_MODEL_ID:-$MODEL_ID_8B_FP8_DEFAULT}"
QWEN_4B_DIR="${BOOKMIND_QWEN3_VL_4B_DIR:-$QWEN_4B_DIR_DEFAULT}"
QWEN_8B_FP8_DIR="${BOOKMIND_QWEN3_VL_8B_FP8_DIR:-$QWEN_8B_FP8_DIR_DEFAULT}"

usage() {
  cat <<'EOF' >&2
Usage: prefetch_online.sh --profile marker|ocr|vlm|all [--sample-pdf /path.pdf] [--sample-image /path.png] [--qwen3vl 4b|8b_fp8|all]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample-pdf)
      SAMPLE_PDF="$2"
      shift 2
      ;;
    --sample-image)
      SAMPLE_IMAGE="$2"
      shift 2
      ;;
    --qwen3vl)
      QWEN3VL="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
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

case "$PROFILE" in
  marker|ocr|vlm|all) ;;
  *)
    echo "Invalid profile: $PROFILE" >&2
    usage
    exit 1
    ;;
esac

case "$QWEN3VL" in
  4b|8b_fp8|all) ;;
  *)
    echo "Invalid --qwen3vl value: $QWEN3VL" >&2
    usage
    exit 1
    ;;
esac

profiles=()
if [[ "$PROFILE" == "all" ]]; then
  profiles=(marker ocr vlm)
else
  profiles=("$PROFILE")
fi

mkdir -p "$WHEEL_DIR" "$MODEL_DIR"

echo "Downloading base wheels into $WHEEL_DIR"
python -m pip download -r "$REQ_DIR/base.txt" -d "$WHEEL_DIR"

for profile in "${profiles[@]}"; do
  case "$profile" in
    marker)
      echo "Downloading marker wheels into $WHEEL_DIR"
      python -m pip download -r "$REQ_DIR/marker.txt" -d "$WHEEL_DIR"
      ;;
    ocr)
      echo "Downloading PaddleOCR wheels into $WHEEL_DIR"
      python -m pip download -r "$REQ_DIR/paddleocr.txt" -d "$WHEEL_DIR"
      ;;
    vlm)
      echo "Downloading VLM wheels into $WHEEL_DIR"
      python -m pip download -r "$REQ_DIR/vlm.txt" -d "$WHEEL_DIR"
      ;;
  esac
done

if [[ " ${profiles[*]} " == *" marker "* ]]; then
  mkdir -p "$MARKER_DIR"
  export XDG_CACHE_HOME="$MARKER_DIR"
  export MARKER_CACHE_DIR="$MARKER_DIR"
  export HF_HOME="$MARKER_DIR"

  if command -v marker_single >/dev/null 2>&1; then
    if [[ -n "$SAMPLE_PDF" && -f "$SAMPLE_PDF" ]]; then
      TMP_OUT="$(mktemp -d)"
      marker_single --pdf "$SAMPLE_PDF" --output "$TMP_OUT" || true
      rm -rf "$TMP_OUT"
    else
      echo "Marker prefetch skipped (provide --sample-pdf /path/to/sample.pdf to prewarm caches)."
    fi
  else
    echo "marker_single not found on PATH; install marker-pdf before prefetching marker assets." >&2
  fi
fi

if [[ " ${profiles[*]} " == *" ocr "* ]]; then
  mkdir -p "$PADDLE_DIR"
  export PADDLEOCR_HOME="$PADDLE_DIR"
  export PADDLE_HOME="$PADDLE_DIR"
  export HOME="$PADDLE_DIR"
  export BOOKMIND_SAMPLE_IMAGE="$SAMPLE_IMAGE"

  python - <<'PY'
import os
from pathlib import Path

sample_image = os.environ.get("BOOKMIND_SAMPLE_IMAGE")

try:
    from paddleocr import PPStructure
except Exception as exc:
    print(f"Unable to import PaddleOCR for warmup: {exc}")
    raise SystemExit(0)

engine = PPStructure(show_log=False)

if sample_image and Path(sample_image).exists():
    try:
        engine(sample_image)
    except Exception as exc:
        print(f"PP-Structure warmup failed: {exc}")
else:
    print("PaddleOCR prefetch initialized (no sample image provided).")
PY
fi

if [[ " ${profiles[*]} " == *" vlm "* ]]; then
  download_qwen3vl() {
    local model_id="$1"
    local target_dir="$2"

    mkdir -p "$target_dir"
    export HF_HOME="$target_dir"
    export HF_HUB_CACHE="$target_dir"
    export TRANSFORMERS_CACHE="$target_dir"

    if command -v huggingface-cli >/dev/null 2>&1; then
      huggingface-cli download "$model_id" --local-dir "$target_dir" --local-dir-use-symlinks False
    elif command -v hf >/dev/null 2>&1; then
      hf download "$model_id" --local-dir "$target_dir"
    else
      echo "huggingface-cli (or hf) not found; install huggingface-hub before downloading Qwen3-VL." >&2
      return 1
    fi
  }

  if [[ "$QWEN3VL" == "4b" || "$QWEN3VL" == "all" ]]; then
    download_qwen3vl "$MODEL_ID_4B" "$QWEN_4B_DIR"
  fi
  if [[ "$QWEN3VL" == "8b_fp8" || "$QWEN3VL" == "all" ]]; then
    download_qwen3vl "$MODEL_ID_8B_FP8" "$QWEN_8B_FP8_DIR"
  fi
fi

echo "Prefetch complete. Bundle at $BUNDLE_DIR"
