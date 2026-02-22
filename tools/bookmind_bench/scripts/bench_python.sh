#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BENCH_PY="$ROOT_DIR/.venv_bookmind_bench_qdrant/bin/python"

if [[ -x "$BENCH_PY" ]]; then
  exec "$BENCH_PY" "$@"
fi

echo "bench_python.sh: bench venv not found at $BENCH_PY; falling back to python3." >&2
echo "Hint: create it with docs/bookmind/BENCH_ENV.md (or create_venv.sh --profile qdrant)." >&2
exec python3 "$@"
