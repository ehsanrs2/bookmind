import sys
from pathlib import Path

# Ensure tests can import local bench modules when run from repo root.
ROOT_DIR = Path(__file__).resolve().parents[3]
BENCH_DIR = ROOT_DIR / "tools" / "bookmind_bench"

for path in (str(ROOT_DIR), str(BENCH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
