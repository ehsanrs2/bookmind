#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LEGACY_VENV_DIR="$ROOT_DIR/.venv_bookmind_bench"
REQ_DIR="$ROOT_DIR/tools/bookmind_bench/requirements"
BUNDLE_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle"
WHEEL_DIR="$BUNDLE_DIR/wheels"

MODE="online"
PROFILE="all"
INSTALL_DEV=0

usage() {
  cat <<'EOF' >&2
Usage: create_venv.sh [--online|--offline|--mode online|offline] --profile marker|ocr|layout|vlm|all [--dev]
Examples:
  create_venv.sh --online --profile ocr
  create_venv.sh --online --profile layout
  create_venv.sh --offline --profile marker
  create_venv.sh --online --profile all --dev
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --online)
      MODE="online"
      shift
      ;;
    --offline)
      MODE="offline"
      shift
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --dev)
      INSTALL_DEV=1
      shift
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

case "$MODE" in
  online|offline) ;;
  *)
    echo "Invalid mode: $MODE" >&2
    usage
    exit 1
    ;;
esac

case "$PROFILE" in
  marker|ocr|layout|vlm|all) ;;
  *)
    echo "Invalid profile: $PROFILE" >&2
    usage
    exit 1
    ;;
esac

if [[ "$MODE" == "offline" ]]; then
  if [[ ! -d "$WHEEL_DIR" ]]; then
    echo "Missing wheel bundle at $WHEEL_DIR" >&2
    exit 1
  fi
  PIP_FLAGS=(--no-index --find-links "$WHEEL_DIR")
else
  PIP_FLAGS=()
fi

PADDLE_GPU_VERSION="${PADDLE_GPU_VERSION:-3.3.0}"
PADDLE_CUDA_INDEX="${PADDLE_CUDA_INDEX:-}"

profiles=()
if [[ "$PROFILE" == "all" ]]; then
  profiles=(marker ocr layout vlm)
else
  profiles=("$PROFILE")
fi

create_profile_env() {
  local profile="$1"
  local venv_dir="$ROOT_DIR/.venv_bookmind_bench_${profile}"
  local req_file=""

  case "$profile" in
    marker) req_file="$REQ_DIR/marker.txt" ;;
    ocr) req_file="$REQ_DIR/paddleocr.txt" ;;
    layout) req_file="$REQ_DIR/layout.txt" ;;
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

  if [[ "$INSTALL_DEV" -eq 1 ]]; then
    python -m pip install "${PIP_FLAGS[@]}" -r "$REQ_DIR/dev.txt"
  fi

  if [[ "$MODE" == "online" && ( "$profile" == "ocr" || "$profile" == "layout" ) ]]; then
    local cuda_ver=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      cuda_ver="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    fi
    if [[ -z "$cuda_ver" ]] && command -v nvcc >/dev/null 2>&1; then
      cuda_ver="$(nvcc --version | awk '/release/{print $NF}' | tr -d 'V,')"
    fi

    if [[ -z "$PADDLE_CUDA_INDEX" ]]; then
      case "$cuda_ver" in
        12.9*) PADDLE_CUDA_INDEX="cu129" ;;
        12.6*) PADDLE_CUDA_INDEX="cu126" ;;
        11.8*) PADDLE_CUDA_INDEX="cu118" ;;
        *) PADDLE_CUDA_INDEX="" ;;
      esac
    fi

    if [[ -n "$PADDLE_CUDA_INDEX" ]]; then
      python -m pip install --upgrade "paddlepaddle-gpu==${PADDLE_GPU_VERSION}" \
        -i "https://www.paddlepaddle.org.cn/packages/stable/${PADDLE_CUDA_INDEX}/"
    else
      echo "CUDA version not detected or unsupported for Paddle ${PADDLE_GPU_VERSION}; keeping requirements pin." >&2
    fi
  fi

  deactivate || true
  echo "Venv ready at $venv_dir (mode: $MODE, profile: $profile)"
}

for profile in "${profiles[@]}"; do
  create_profile_env "$profile"
done

if [[ -d "$LEGACY_VENV_DIR" ]]; then
  echo "Legacy venv detected at $LEGACY_VENV_DIR (kept for compatibility)." >&2
fi
