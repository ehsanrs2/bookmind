#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REQ_DIR="$ROOT_DIR/tools/bookmind_bench/requirements"
BUNDLE_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle"
WHEEL_DIR="$BUNDLE_DIR/wheels"
LAYOUT_WHEEL_DIR="$WHEEL_DIR/layout"
QDRANT_WHEEL_DIR="$WHEEL_DIR/qdrant"
MODEL_DIR="$BUNDLE_DIR/models"

PROFILE="all"

usage() {
  cat <<'EOF' >&2
Usage: install_offline.sh --profile marker|ocr|layout|vlm|qdrant|all
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
  marker|ocr|layout|vlm|qdrant|all) ;;
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
  profiles=(marker ocr layout vlm qdrant)
else
  profiles=("$PROFILE")
fi

PIP_FLAGS=(--no-index --find-links "$WHEEL_DIR")

select_layout_python() {
  if command -v python3.10 >/dev/null 2>&1; then
    echo "python3.10"
    return 0
  fi
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
    return 0
  fi
  echo "Layout profile requires python3.10 or python3.11. Install one of them and retry." >&2
  exit 1
}

install_profile() {
  local profile="$1"
  local venv_dir="$ROOT_DIR/.venv_bookmind_bench_${profile}"
  local req_file=""
  local profile_pip_flags=("${PIP_FLAGS[@]}")
  local python_bin="python3"

  case "$profile" in
    marker) req_file="$REQ_DIR/marker.txt" ;;
    ocr) req_file="$REQ_DIR/paddleocr.txt" ;;
    layout) req_file="$REQ_DIR/layout.txt" ;;
    vlm) req_file="$REQ_DIR/vlm.txt" ;;
    qdrant) req_file="$REQ_DIR/qdrant.txt" ;;
  esac

  if [[ "$profile" == "layout" ]]; then
    if [[ ! -d "$LAYOUT_WHEEL_DIR" ]]; then
      echo "Missing layout offline wheel bundle at $LAYOUT_WHEEL_DIR" >&2
      exit 1
    fi
    if ! ls "$LAYOUT_WHEEL_DIR"/detectron2-*.whl >/dev/null 2>&1; then
      echo "Missing detectron2 wheel in $LAYOUT_WHEEL_DIR. Run prefetch_online.sh --profile layout on a network-enabled machine first." >&2
      exit 1
    fi
    profile_pip_flags=(--no-index --find-links "$LAYOUT_WHEEL_DIR")
    python_bin="$(select_layout_python)"
  fi
  if [[ "$profile" == "qdrant" ]]; then
    if [[ ! -d "$QDRANT_WHEEL_DIR" ]]; then
      echo "Missing qdrant offline wheel bundle at $QDRANT_WHEEL_DIR" >&2
      exit 1
    fi
    profile_pip_flags=(--no-index --find-links "$QDRANT_WHEEL_DIR")
  fi

  if [[ ! -d "$venv_dir" ]]; then
    "$python_bin" -m venv "$venv_dir"
  fi

  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"
  if [[ "$profile" == "layout" ]]; then
    layout_py_minor="$(python -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')"
    if [[ "$layout_py_minor" != "3.10" && "$layout_py_minor" != "3.11" ]]; then
      deactivate || true
      echo "Layout venv at $venv_dir uses Python $layout_py_minor; recreate with python3.10 or python3.11." >&2
      exit 1
    fi
  fi
  python -m pip install "${profile_pip_flags[@]}" --upgrade pip setuptools wheel
  python -m pip install "${profile_pip_flags[@]}" -r "$REQ_DIR/base.txt"
  python -m pip install "${profile_pip_flags[@]}" -r "$req_file"
  if [[ "$profile" == "layout" ]]; then
    python -m pip install "${profile_pip_flags[@]}" detectron2
  fi
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
if [[ " ${profiles[*]} " == *" layout "* ]]; then
  echo "  export BOOKMIND_LAYOUT_MODEL_DIR=\"$MODEL_DIR/layoutparser_publaynet\""
  echo "  export PADDLEOCR_HOME=\"$MODEL_DIR/paddleocr\""
  echo "  export PADDLE_HOME=\"$MODEL_DIR/paddleocr\""
  echo "  export HOME=\"$MODEL_DIR/paddleocr\""
fi
if [[ " ${profiles[*]} " == *" vlm "* ]]; then
  echo "  export HF_HOME=\"$MODEL_DIR/qwen3_vl/4b\""
  echo "  export HF_HUB_CACHE=\"$MODEL_DIR/qwen3_vl/4b\""
  echo "  export TRANSFORMERS_CACHE=\"$MODEL_DIR/qwen3_vl/4b\""
fi
if [[ " ${profiles[*]} " == *" qdrant "* ]]; then
  echo "  export HF_HOME=\"$MODEL_DIR/embeddings/all-MiniLM-L6-v2\""
  echo "  export HF_HUB_CACHE=\"$MODEL_DIR/embeddings/all-MiniLM-L6-v2\""
  echo "  export TRANSFORMERS_CACHE=\"$MODEL_DIR/embeddings/all-MiniLM-L6-v2\""
  echo "  export HF_HUB_OFFLINE=1"
  echo "  export TRANSFORMERS_OFFLINE=1"
fi
