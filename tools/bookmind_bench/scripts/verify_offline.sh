#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv_bookmind_bench"
BUNDLE_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle"
MODEL_DIR="$BUNDLE_DIR/models"

SAMPLE_PDF=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdf)
      SAMPLE_PDF="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--pdf /path/to/sample.pdf]" >&2
      exit 1
      ;;
  esac
 done

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing venv at $VENV_DIR. Run install_offline.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$MODEL_DIR/qwen3_vl"
export HF_HUB_CACHE="$MODEL_DIR/qwen3_vl"
export TRANSFORMERS_CACHE="$MODEL_DIR/qwen3_vl"
export XDG_CACHE_HOME="$MODEL_DIR/marker"
export MARKER_CACHE_DIR="$MODEL_DIR/marker"
export PADDLEOCR_HOME="$MODEL_DIR/paddleocr"
export PADDLE_HOME="$MODEL_DIR/paddleocr"

python "$ROOT_DIR/tools/bookmind_bench/run.py" render --help
python "$ROOT_DIR/tools/bookmind_bench/run.py" marker --help
python "$ROOT_DIR/tools/bookmind_bench/run.py" paddleocr --help

if [[ -n "$SAMPLE_PDF" ]]; then
  if [[ ! -f "$SAMPLE_PDF" ]]; then
    echo "Sample PDF not found: $SAMPLE_PDF" >&2
    exit 1
  fi
  OUT_DIR="$(mktemp -d)"
  python "$ROOT_DIR/tools/bookmind_bench/run.py" marker --pdf "$SAMPLE_PDF" --out "$OUT_DIR"
  rm -rf "$OUT_DIR"
else
  echo "No --pdf provided; marker smoke run skipped."
fi

echo "Offline verification complete."
