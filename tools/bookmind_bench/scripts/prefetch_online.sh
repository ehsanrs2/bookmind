#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REQ_DIR="$ROOT_DIR/tools/bookmind_bench/requirements"
BUNDLE_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle"
WHEEL_DIR="$BUNDLE_DIR/wheels"
LAYOUT_WHEEL_DIR="$WHEEL_DIR/layout"
MODEL_DIR="$BUNDLE_DIR/models"
MARKER_DIR="$MODEL_DIR/marker"
PADDLE_DIR="$MODEL_DIR/paddleocr"
LAYOUT_DIR="$MODEL_DIR/layoutparser_publaynet"
QWEN_DIR_BASE="$MODEL_DIR/qwen3_vl"
EMBED_DIR="$MODEL_DIR/embeddings"
QDRANT_WHEEL_DIR="$WHEEL_DIR/qdrant"
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
Usage: prefetch_online.sh --profile marker|ocr|layout|vlm|qdrant|all [--sample-pdf /path.pdf] [--sample-image /path.png] [--qwen3vl 4b|8b_fp8|all]
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
  marker|ocr|layout|vlm|qdrant|all) ;;
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
  profiles=(marker ocr layout vlm qdrant)
else
  profiles=("$PROFILE")
fi

if [[ " ${profiles[*]} " == *" layout "* ]]; then
  if ! python - <<'PY'
import socket

targets = [("pypi.org", 443), ("github.com", 443)]
for host, port in targets:
    try:
        socket.gethostbyname(host)
        with socket.create_connection((host, port), timeout=4):
            pass
    except Exception:
        raise SystemExit(1)
raise SystemExit(0)
PY
  then
    echo "Network required for online prefetch. Run this step on a network-enabled machine, then copy offline_bundle/ to offline system." >&2
    exit 1
  fi
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
    qdrant)
      mkdir -p "$QDRANT_WHEEL_DIR"
      echo "Downloading Qdrant wheels into $QDRANT_WHEEL_DIR"
      python -m pip download -r "$REQ_DIR/base.txt" -r "$REQ_DIR/qdrant.txt" -d "$QDRANT_WHEEL_DIR"
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

if [[ " ${profiles[*]} " == *" layout "* ]]; then
  mkdir -p "$LAYOUT_WHEEL_DIR"

  "$ROOT_DIR/tools/bookmind_bench/scripts/create_venv.sh" --online --profile layout
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv_bookmind_bench_layout/bin/activate"

  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "$REQ_DIR/base.txt"
  python -m pip install -r "$REQ_DIR/layout.txt"
  python -c "import numpy; print(f'numpy in layout venv: {numpy.__version__}')"

  "$ROOT_DIR/tools/bookmind_bench/scripts/build_detectron2_wheel.sh"
  python -m pip install --no-index --find-links "$LAYOUT_WHEEL_DIR" --no-deps detectron2
  python -m pip download -d "$LAYOUT_WHEEL_DIR" -r "$REQ_DIR/base.txt" -r "$REQ_DIR/layout.txt"
  python -m pip download -d "$LAYOUT_WHEEL_DIR" --find-links "$LAYOUT_WHEEL_DIR" detectron2
  if ! ls "$LAYOUT_WHEEL_DIR"/numpy-*.whl >/dev/null 2>&1; then
    echo "Layout wheelhouse incomplete: numpy wheel not found in $LAYOUT_WHEEL_DIR" >&2
    exit 1
  fi

  mkdir -p "$LAYOUT_DIR"
  export BOOKMIND_LAYOUT_MODEL_DIR="$LAYOUT_DIR"
  export FVCORE_CACHE="$LAYOUT_DIR"
  export DETECTRON2_DATASETS="$LAYOUT_DIR"

  python - <<'PY'
import json
import os
from pathlib import Path

layout_dir = Path(os.environ["BOOKMIND_LAYOUT_MODEL_DIR"])
layout_dir.mkdir(parents=True, exist_ok=True)

config_candidates = sorted(layout_dir.rglob("*.yaml")) + sorted(layout_dir.rglob("*.yml"))
weight_candidates = sorted(layout_dir.rglob("*.pth")) + sorted(layout_dir.rglob("*.pkl"))

if config_candidates and weight_candidates:
    print(f"Layout assets already present at {layout_dir}")
    raise SystemExit(0)

try:
    import layoutparser as lp
except Exception as exc:
    print(f"Unable to import layoutparser for layout prefetch: {exc}")
    print("Install layout profile venv and rerun to prewarm PubLayNet assets.")
    raise SystemExit(0)

try:
    lp.Detectron2LayoutModel(
        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    )
except Exception as exc:
    print(f"Layout prefetch warmup failed: {exc}")
    print(
        "Place local Detectron2 PubLayNet config (.yaml) and weights (.pth) under "
        f"{layout_dir} for offline layout runs."
    )
    raise SystemExit(0)

config_candidates = sorted(layout_dir.rglob("*.yaml")) + sorted(layout_dir.rglob("*.yml"))
weight_candidates = sorted(layout_dir.rglob("*.pth")) + sorted(layout_dir.rglob("*.pkl"))

manifest = {
    "config_files": [str(p) for p in config_candidates],
    "weight_files": [str(p) for p in weight_candidates],
}
(layout_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Layout prefetch complete at {layout_dir}")
PY
  deactivate || true
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

if [[ " ${profiles[*]} " == *" qdrant "* ]]; then
  EMBED_MODEL_ID_DEFAULT="sentence-transformers/all-MiniLM-L6-v2"
  EMBED_MODEL_ID="${BOOKMIND_EMBED_MODEL_ID:-$EMBED_MODEL_ID_DEFAULT}"
  EMBED_MODEL_DIR="${BOOKMIND_EMBED_MODEL_DIR:-$EMBED_DIR/all-MiniLM-L6-v2}"

  "$ROOT_DIR/tools/bookmind_bench/scripts/create_venv.sh" --online --profile qdrant

  mkdir -p "$EMBED_MODEL_DIR"
  export HF_HOME="$EMBED_MODEL_DIR"
  export HF_HUB_CACHE="$EMBED_MODEL_DIR"
  export TRANSFORMERS_CACHE="$EMBED_MODEL_DIR"

  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$EMBED_MODEL_ID" --local-dir "$EMBED_MODEL_DIR" --local-dir-use-symlinks False
  elif command -v hf >/dev/null 2>&1; then
    hf download "$EMBED_MODEL_ID" --local-dir "$EMBED_MODEL_DIR"
  else
    echo "huggingface-cli (or hf) not found; install huggingface-hub before downloading embedding model." >&2
    exit 1
  fi
fi

echo "Prefetch complete. Bundle at $BUNDLE_DIR"
