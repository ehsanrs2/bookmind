#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REQ_DIR="$ROOT_DIR/tools/bookmind_bench/requirements"
BUNDLE_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle"
WHEEL_DIR="$BUNDLE_DIR/wheels"
MODEL_DIR="$BUNDLE_DIR/models"

PROFILE="all"

usage() {
  cat <<'EOF' >&2
Usage: install_offline.sh --profile marker|ocr|vlm|all
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

if [[ ! -d "$WHEEL_DIR" ]]; then
  echo "Missing offline wheel bundle at $WHEEL_DIR" >&2
  exit 1
fi

profiles=()
if [[ "$PROFILE" == "all" ]]; then
  profiles=(marker ocr vlm)
else
  profiles=("$PROFILE")
fi

PIP_FLAGS=(--no-index --find-links "$WHEEL_DIR")

install_profile() {
  local profile="$1"
  local venv_dir="$ROOT_DIR/.venv_bookmind_bench_${profile}"
  local req_file=""

  case "$profile" in
    marker) req_file="$REQ_DIR/marker.txt" ;;
    ocr) req_file="$REQ_DIR/paddleocr.txt" ;;
    vlm) req_file="$REQ_DIR/vlm.txt" ;;
  esac

  if [[ ! -d "$venv_dir" ]]; then
    python3 -m venv "$venv_dir"
  fi

  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"
  python -m pip install "${PIP_FLAGS[@]}" --upgrade pip setuptools wheel
  python -m pip install "${PIP_FLAGS[@]}" -r "$REQ_DIR/base.txt"
  python -m pip install "${PIP_FLAGS[@]}" -r "$req_file"
  if [[ "$profile" == "vlm" ]]; then
    if python -m pip show vllm-flash-attn >/dev/null 2>&1; then
      python -m pip uninstall -y vllm-flash-attn
    fi
  fi
  deactivate || true

  echo "Offline install complete. Venv: $venv_dir (profile: $profile)"
}

for profile in "${profiles[@]}"; do
  install_profile "$profile"
done

echo ""
echo "Recommended cache environment variables:"
if [[ " ${profiles[*]} " == *" marker "* ]]; then
  echo "  export XDG_CACHE_HOME=\"$MODEL_DIR/marker\""
  echo "  export MARKER_CACHE_DIR=\"$MODEL_DIR/marker\""
fi
if [[ " ${profiles[*]} " == *" ocr "* ]]; then
  echo "  export PADDLEOCR_HOME=\"$MODEL_DIR/paddleocr\""
  echo "  export PADDLE_HOME=\"$MODEL_DIR/paddleocr\""
  echo "  export HOME=\"$MODEL_DIR/paddleocr\""
fi
if [[ " ${profiles[*]} " == *" vlm "* ]]; then
  echo "  export HF_HOME=\"$MODEL_DIR/qwen3_vl\""
  echo "  export HF_HUB_CACHE=\"$MODEL_DIR/qwen3_vl\""
  echo "  export TRANSFORMERS_CACHE=\"$MODEL_DIR/qwen3_vl\""
fi
