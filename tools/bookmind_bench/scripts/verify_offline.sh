#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LEGACY_VENV_DIR="$ROOT_DIR/.venv_bookmind_bench"
BUNDLE_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle"
MODEL_DIR="$BUNDLE_DIR/models"

SAMPLE_PDF=""
SAMPLE_IMAGE=""
PROFILE="all"

usage() {
  cat <<'EOF' >&2
Usage: verify_offline.sh --profile marker|ocr|vlm|all [--pdf /path/to/sample.pdf] [--image /path/to/sample.png]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdf)
      SAMPLE_PDF="$2"
      shift 2
      ;;
    --image)
      SAMPLE_IMAGE="$2"
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

ensure_cache_dir() {
  local dir="$1"
  local label="$2"
  if [[ ! -d "$dir" || -z "$(ls -A "$dir" 2>/dev/null)" ]]; then
    echo "Missing offline cache for $label at $dir. Run prefetch_online.sh and copy offline_bundle/ first." >&2
    exit 1
  fi
}

resolve_venv() {
  local profile="$1"
  local venv_dir="$ROOT_DIR/.venv_bookmind_bench_${profile}"
  if [[ -d "$venv_dir" ]]; then
    echo "$venv_dir"
    return 0
  fi
  if [[ -d "$LEGACY_VENV_DIR" ]]; then
    echo "Warning: using legacy venv at $LEGACY_VENV_DIR for profile $profile." >&2
    echo "$LEGACY_VENV_DIR"
    return 0
  fi
  echo "Missing venv for profile $profile. Run install_offline.sh first." >&2
  exit 1
}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

for profile in "${profiles[@]}"; do
  venv_dir="$(resolve_venv "$profile")"
  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"

  python "$ROOT_DIR/tools/bookmind_bench/run.py" render --help

  case "$profile" in
    marker)
      ensure_cache_dir "$MODEL_DIR/marker" "marker"
      export XDG_CACHE_HOME="$MODEL_DIR/marker"
      export MARKER_CACHE_DIR="$MODEL_DIR/marker"
      export HF_HOME="$MODEL_DIR/marker"
      python "$ROOT_DIR/tools/bookmind_bench/run.py" marker --help
      if [[ -n "$SAMPLE_PDF" ]]; then
        if [[ ! -f "$SAMPLE_PDF" ]]; then
          echo "Sample PDF not found: $SAMPLE_PDF" >&2
          exit 1
        fi
        OUT_DIR="$(mktemp -d)"
        python "$ROOT_DIR/tools/bookmind_bench/run.py" marker --pdf "$SAMPLE_PDF" --out "$OUT_DIR" --pages "1"
        rm -rf "$OUT_DIR"
      else
        echo "No --pdf provided; marker smoke run skipped."
      fi
      ;;
    ocr)
      ensure_cache_dir "$MODEL_DIR/paddleocr" "PaddleOCR"
      export PADDLEOCR_HOME="$MODEL_DIR/paddleocr"
      export PADDLE_HOME="$MODEL_DIR/paddleocr"
      python "$ROOT_DIR/tools/bookmind_bench/run.py" paddleocr --help
      if [[ -n "$SAMPLE_IMAGE" ]]; then
        if [[ ! -f "$SAMPLE_IMAGE" ]]; then
          echo "Sample image not found: $SAMPLE_IMAGE" >&2
          exit 1
        fi
        python "$ROOT_DIR/tools/bookmind_bench/run.py" paddleocr-smoke --image "$SAMPLE_IMAGE"
      else
        echo "No --image provided; PaddleOCR smoke run skipped."
      fi
      ;;
    vlm)
      ensure_cache_dir "$MODEL_DIR/qwen3_vl" "Qwen3-VL"
      export HF_HOME="$MODEL_DIR/qwen3_vl"
      export HF_HUB_CACHE="$MODEL_DIR/qwen3_vl"
      export TRANSFORMERS_CACHE="$MODEL_DIR/qwen3_vl"
      ;;
  esac

  deactivate || true
done

echo "Offline verification complete."
