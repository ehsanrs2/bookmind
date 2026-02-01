#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv_bookmind_bench"
REQ_DIR="$ROOT_DIR/tools/bookmind_bench/requirements"
BUNDLE_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle"
WHEEL_DIR="$BUNDLE_DIR/wheels"
MODEL_DIR="$BUNDLE_DIR/models"

if [[ ! -d "$WHEEL_DIR" ]]; then
  echo "Missing offline wheel bundle at $WHEEL_DIR" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export XDG_CACHE_HOME="$MODEL_DIR/marker"
export MARKER_CACHE_DIR="$MODEL_DIR/marker"
export HF_HOME="$MODEL_DIR/qwen3_vl"
export HF_HUB_CACHE="$MODEL_DIR/qwen3_vl"
export TRANSFORMERS_CACHE="$MODEL_DIR/qwen3_vl"
export PADDLEOCR_HOME="$MODEL_DIR/paddleocr"
export PADDLE_HOME="$MODEL_DIR/paddleocr"

PIP_FLAGS=(--no-index --find-links "$WHEEL_DIR")
python -m pip install "${PIP_FLAGS[@]}" -r "$REQ_DIR/base.txt"
python -m pip install "${PIP_FLAGS[@]}" -r "$REQ_DIR/marker.txt"
python -m pip install "${PIP_FLAGS[@]}" -r "$REQ_DIR/paddleocr.txt"
python -m pip install "${PIP_FLAGS[@]}" -r "$REQ_DIR/vlm.txt"

echo "Offline install complete. Venv: $VENV_DIR"
