#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAYOUT_WHEEL_DIR="$ROOT_DIR/tools/bookmind_bench/offline_bundle/wheels/layout"
DETECTRON2_REPO_URL="https://github.com/facebookresearch/detectron2.git"
# Pinned detectron2 commit on main branch for reproducible source builds.
DETECTRON2_COMMIT="${BOOKMIND_DETECTRON2_COMMIT:-fd27788}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate the layout venv first, then rerun this script." >&2
  echo "Example: source .venv_bookmind_bench_layout/bin/activate" >&2
  exit 1
fi

python_minor="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python_dev_pkg="python${python_minor}-dev"
if [[ "$python_minor" != "3.10" && "$python_minor" != "3.11" ]]; then
  echo "Layout wheel build requires Python 3.10 or 3.11, found $python_minor in active venv." >&2
  exit 1
fi

print_prereq_hint() {
  echo "Install prerequisites (Ubuntu/Debian):" >&2
  echo "  sudo apt-get update && sudo apt-get install -y build-essential cmake ninja-build git $python_dev_pkg" >&2
}

if ! command -v gcc >/dev/null 2>&1; then
  echo "Missing system dependency: gcc" >&2
  print_prereq_hint
  exit 1
fi
if ! command -v g++ >/dev/null 2>&1; then
  echo "Missing system dependency: g++" >&2
  print_prereq_hint
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "Missing system dependency: cmake" >&2
  print_prereq_hint
  exit 1
fi
if ! command -v ninja >/dev/null 2>&1; then
  echo "Missing system dependency: ninja (install: sudo apt-get update && sudo apt-get install -y ninja-build)" >&2
  print_prereq_hint
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "Missing system dependency: git" >&2
  print_prereq_hint
  exit 1
fi

python - <<'PY'
import pathlib
import sys
import sysconfig

py_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
include_dir = sysconfig.get_config_var("INCLUDEPY")
if not include_dir:
    raise SystemExit(
        f"Python development headers not found (INCLUDEPY empty). Install python{py_minor}-dev."
    )
header = pathlib.Path(include_dir) / "Python.h"
if not header.exists():
    raise SystemExit(
        f"Python development headers missing at {header}. Install python{py_minor}-dev."
    )
PY

python -m pip install --upgrade pip wheel setuptools pybind11

mkdir -p "$LAYOUT_WHEEL_DIR"
rm -f "$LAYOUT_WHEEL_DIR"/detectron2-*.whl

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

git clone --depth 1 "$DETECTRON2_REPO_URL" "$workdir/detectron2"
(
  cd "$workdir/detectron2"
  git fetch --depth 1 origin "$DETECTRON2_COMMIT"
  git checkout "$DETECTRON2_COMMIT"
)

python -m pip wheel --no-deps --wheel-dir "$LAYOUT_WHEEL_DIR" "$workdir/detectron2"

wheel_path="$(ls -1 "$LAYOUT_WHEEL_DIR"/detectron2-*.whl | head -n 1)"
if [[ -z "$wheel_path" ]]; then
  echo "Detectron2 wheel build failed: no wheel produced in $LAYOUT_WHEEL_DIR" >&2
  exit 1
fi

echo "Built Detectron2 wheel: $wheel_path"
