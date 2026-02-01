#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv_bookmind_bench"
REQ_DIR="$ROOT_DIR/tools/bookmind_bench/requirements"
BUNDLE_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle"
WHEEL_DIR="$BUNDLE_DIR/wheels"

MODE="online"
if [[ "${1:-}" == "--offline" ]]; then
  MODE="offline"
elif [[ "${1:-}" == "--online" || -z "${1:-}" ]]; then
  MODE="online"
else
  echo "Usage: $0 [--online|--offline]" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

REQ_FILES=(
  "$REQ_DIR/base.txt"
  "$REQ_DIR/marker.txt"
  "$REQ_DIR/paddleocr.txt"
  "$REQ_DIR/vlm.txt"
)

if [[ "$MODE" == "offline" ]]; then
  if [[ ! -d "$WHEEL_DIR" ]]; then
    echo "Missing wheel bundle at $WHEEL_DIR" >&2
    exit 1
  fi
  PIP_FLAGS=(--no-index --find-links "$WHEEL_DIR")
else
  PIP_FLAGS=()
fi

python -m pip install "${PIP_FLAGS[@]}" -r "$REQ_DIR/base.txt"
python -m pip install "${PIP_FLAGS[@]}" -r "$REQ_DIR/marker.txt"
python -m pip install "${PIP_FLAGS[@]}" -r "$REQ_DIR/paddleocr.txt"
python -m pip install "${PIP_FLAGS[@]}" -r "$REQ_DIR/vlm.txt"

echo "Venv ready at $VENV_DIR (mode: $MODE)"
