#!/usr/bin/env bash
set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MODEL_TAG="${MODEL_TAG:-qwen3-vl:latest}"

cat <<EOF
Offline Ollama backend helper (no downloads, no pulls).

This script does not start or pull models automatically.
It only checks your local Ollama API and whether the expected model tag exists.

Expected:
  OLLAMA_URL=${OLLAMA_URL}
  MODEL_TAG=${MODEL_TAG}

Sanity check API tags:
  curl -fsS ${OLLAMA_URL}/api/tags
EOF

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for sanity checks." >&2
  exit 1
fi

if ! tags_json="$(curl -fsS "${OLLAMA_URL}/api/tags")"; then
  echo "Unable to reach Ollama at ${OLLAMA_URL}. Start Ollama first." >&2
  exit 1
fi

if printf "%s" "$tags_json" | grep -q "\"name\":\"${MODEL_TAG}\""; then
  echo "Model tag found: ${MODEL_TAG}"
  echo "You can now run:"
  cat <<EOF
python tools/bookmind_bench/run.py vlm-layout \\
  --backend ollama \\
  --ollama_url ${OLLAMA_URL} \\
  --ollama_model ${MODEL_TAG} \\
  --imgdir <IMG_DIR> \\
  --out <RUN_DIR>
EOF
else
  echo "Model tag not found: ${MODEL_TAG}" >&2
  echo "Available tags from ${OLLAMA_URL}/api/tags:" >&2
  printf "%s\n" "$tags_json" >&2
  echo "This helper is offline-first and will not pull models for you." >&2
  exit 1
fi
