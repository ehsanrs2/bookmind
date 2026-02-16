#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-sentence-transformers/all-MiniLM-L6-v2}"
CACHE_DIR="${2:-}"

if [[ -n "${CACHE_DIR}" ]]; then
  export HF_HOME="${CACHE_DIR}"
  export HF_HUB_CACHE="${CACHE_DIR}"
fi

python - "$MODEL" "$CACHE_DIR" <<'PY'
import sys

model_ref = sys.argv[1]
cache_dir = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

from sentence_transformers import SentenceTransformer

kwargs = {}
if cache_dir:
    kwargs["cache_folder"] = cache_dir
SentenceTransformer(model_ref, **kwargs)
print(f"Prefetched embedding model: {model_ref}")
if cache_dir:
    print(f"Cache dir: {cache_dir}")
PY
