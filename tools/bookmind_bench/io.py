"""IO helpers for Bookmind benchmark runs (offline)."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write JSON objects to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append a single JSON row to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")


@contextmanager
def timer(label: Optional[str] = None) -> Iterator[Dict[str, float]]:
    """Context manager returning a dict with elapsed time in seconds."""
    start = time.perf_counter()
    data: Dict[str, float] = {}
    try:
        yield data
    finally:
        data["elapsed_s"] = time.perf_counter() - start
        if label:
            data["label"] = label


def now_ms() -> int:
    """Current time in milliseconds since epoch."""
    return int(time.time() * 1000)


def durations_to_ms(durations: Iterable[float]) -> List[int]:
    """Convert seconds to rounded milliseconds."""
    return [int(round(d * 1000)) for d in durations]
