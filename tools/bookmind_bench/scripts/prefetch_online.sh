#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REQ_DIR="$ROOT_DIR/tools/bookmind_bench/requirements"
BUNDLE_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle"
WHEEL_DIR="$BUNDLE_DIR/wheels"
MODEL_DIR="$BUNDLE_DIR/models"
MARKER_DIR="$MODEL_DIR/marker"
PADDLE_DIR="$MODEL_DIR/paddleocr"
QWEN_DIR_DEFAULT="$MODEL_DIR/qwen3_vl"

SAMPLE_PDF=""
SAMPLE_IMAGE=""
PROFILE="all"
MODEL_ID_DEFAULT="Qwen/Qwen3-VL-8B-Instruct"
MODEL_ID="${BOOKMIND_QWEN3_VL_MODEL_ID:-$MODEL_ID_DEFAULT}"
QWEN_DIR="${BOOKMIND_QWEN3_VL_DIR:-$QWEN_DIR_DEFAULT}"

usage() {
  cat <<'EOF' >&2
Usage: prefetch_online.sh --profile marker|ocr|vlm|all [--sample-pdf /path.pdf] [--sample-image /path.png] [--qwen-model MODEL_ID]
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
    --qwen-model)
      MODEL_ID="$2"
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
  mkdir -p "$QWEN_DIR"
  export HF_HOME="$QWEN_DIR"
  export HF_HUB_CACHE="$QWEN_DIR"
  export TRANSFORMERS_CACHE="$QWEN_DIR"

  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_ID" --local-dir "$QWEN_DIR" --local-dir-use-symlinks False
  elif command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_ID" --local-dir "$QWEN_DIR" --local-dir-use-symlinks False
  else
    echo "huggingface-cli (or hf) not found; install huggingface-hub before downloading Qwen3-VL." >&2
  fi
fi

echo "Prefetch complete. Bundle at $BUNDLE_DIR"
