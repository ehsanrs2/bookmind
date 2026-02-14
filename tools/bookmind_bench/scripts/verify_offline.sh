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
Usage: verify_offline.sh --profile marker|ocr|layout|vlm|qdrant|all [--pdf /path/to/sample.pdf] [--image /path/to/sample.png]
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
  marker|ocr|layout|vlm|qdrant|all) ;;
  *)
    echo "Invalid profile: $PROFILE" >&2
    usage
    exit 1
    ;;
esac

profiles=()
if [[ "$PROFILE" == "all" ]]; then
  profiles=(marker ocr layout vlm qdrant)
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
      export HOME="$MODEL_DIR/paddleocr"
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
    layout)
      ensure_cache_dir "$MODEL_DIR/layoutparser_publaynet" "LayoutParser PubLayNet"
      ensure_cache_dir "$MODEL_DIR/paddleocr" "PaddleOCR"
      export BOOKMIND_LAYOUT_MODEL_DIR="$MODEL_DIR/layoutparser_publaynet"
      export PADDLEOCR_HOME="$MODEL_DIR/paddleocr"
      export PADDLE_HOME="$MODEL_DIR/paddleocr"
      export HOME="$MODEL_DIR/paddleocr"
      layout_py_minor="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      if [[ "$layout_py_minor" != "3.10" && "$layout_py_minor" != "3.11" ]]; then
        echo "Layout profile requires a Python 3.10/3.11 venv, found $layout_py_minor in $venv_dir." >&2
        exit 1
      fi
      python -c "import numpy; import detectron2; import layoutparser; print('OK imports')"
      python "$ROOT_DIR/tools/bookmind_bench/run.py" layout --help
      if [[ -n "$SAMPLE_IMAGE" ]]; then
        if [[ ! -f "$SAMPLE_IMAGE" ]]; then
          echo "Sample image not found: $SAMPLE_IMAGE" >&2
          exit 1
        fi
        TMP_DIR="$(mktemp -d)"
        cp "$SAMPLE_IMAGE" "$TMP_DIR/page_0001.png"
        OUT_DIR="$(mktemp -d)"
        python "$ROOT_DIR/tools/bookmind_bench/run.py" layout --imgdir "$TMP_DIR" --out "$OUT_DIR" --max_regions_per_page 1
        rm -rf "$TMP_DIR" "$OUT_DIR"
      else
        echo "No --image provided; layout smoke run skipped."
      fi
      ;;
    vlm)
      VLM_DIR_DEFAULT="$MODEL_DIR/qwen3_vl/4b"
      VLM_DIR="${BOOKMIND_QWEN3_VL_DIR:-$VLM_DIR_DEFAULT}"
      ensure_cache_dir "$VLM_DIR" "Qwen3-VL"
      export HF_HOME="$VLM_DIR"
      export HF_HUB_CACHE="$VLM_DIR"
      export TRANSFORMERS_CACHE="$VLM_DIR"
      ;;
    qdrant)
      EMBED_DIR_DEFAULT="$MODEL_DIR/embeddings/all-MiniLM-L6-v2"
      EMBED_DIR="${BOOKMIND_EMBED_MODEL_DIR:-$EMBED_DIR_DEFAULT}"
      ensure_cache_dir "$EMBED_DIR" "embedding model"
      export HF_HOME="$EMBED_DIR"
      export HF_HUB_CACHE="$EMBED_DIR"
      export TRANSFORMERS_CACHE="$EMBED_DIR"
      python -c "import qdrant_client; import sentence_transformers; print('OK imports')"
      export BOOKMIND_VERIFY_EMBED_DIR="$EMBED_DIR"
      python - <<'PY'
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(os.environ["BOOKMIND_VERIFY_EMBED_DIR"])
vec = model.encode(["offline smoke"], normalize_embeddings=True)
assert len(vec) == 1
assert len(vec[0]) > 0
print("OK embedding smoke")
PY
      ;;
  esac

  deactivate || true
done

echo "Offline verification complete."
