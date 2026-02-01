"""Marker engine integration for Bookmind offline benchmarks."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PAGE_RE = re.compile(r"(?:page|pg)[_\- ]*(\d+)", re.IGNORECASE)
_MARKER_PAGE_BREAK_RE = re.compile(r"^\{(\d+)\}-+")


@dataclass(frozen=True)
class MarkerPage:
    page: int
    content: str
    content_type: str
    timing_ms: Optional[int] = None


def _parse_page_num_from_name(path: Path) -> Optional[int]:
    match = _PAGE_RE.search(path.stem)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _extract_timing_ms(obj: Dict[str, Any]) -> Optional[int]:
    for key in (
        "timing_ms",
        "elapsed_ms",
        "duration_ms",
        "time_ms",
        "latency_ms",
    ):
        if key in obj and obj[key] is not None:
            try:
                return int(round(float(obj[key])))
            except (TypeError, ValueError):
                pass

    for key in ("timing_s", "elapsed_s", "duration_s", "time_s"):
        if key in obj and obj[key] is not None:
            try:
                return int(round(float(obj[key]) * 1000))
            except (TypeError, ValueError):
                pass

    return None


def _extract_page_num(obj: Dict[str, Any], index: int) -> int:
    for key in ("page", "page_num", "page_number"):
        if key in obj and obj[key] is not None:
            try:
                value = int(obj[key])
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass

    for key in ("page_index", "pageIndex"):
        if key in obj and obj[key] is not None:
            try:
                value = int(obj[key])
                return value + 1
            except (TypeError, ValueError):
                pass

    return index + 1


def _extract_text_and_type(
    obj: Any, output_format: Optional[str]
) -> Tuple[str, str]:
    if isinstance(obj, str):
        default_type = "page_md" if output_format == "markdown" else "ocr_text"
        return obj, default_type

    if not isinstance(obj, dict):
        return str(obj), "ocr_text"

    for key in ("markdown", "md", "page_md"):
        value = obj.get(key)
        if value:
            return str(value), "page_md"

    if "content" in obj:
        content = obj.get("content")
        if isinstance(content, dict) and content.get("text"):
            return str(content.get("text")), "ocr_text"
        if isinstance(content, str) and content:
            return str(content), "ocr_text"

    for key in ("text", "ocr_text"):
        value = obj.get(key)
        if value:
            return str(value), "ocr_text"

    if obj:
        return json.dumps(obj, ensure_ascii=True), "ocr_text"

    return "", "ocr_text"


def _pages_from_list(
    data: List[Any], output_format: Optional[str]
) -> List[MarkerPage]:
    pages: List[MarkerPage] = []
    for index, item in enumerate(data):
        if isinstance(item, dict):
            page_num = _extract_page_num(item, index)
            timing_ms = _extract_timing_ms(item)
            text, content_type = _extract_text_and_type(item, output_format)
        else:
            page_num = index + 1
            timing_ms = None
            text, content_type = _extract_text_and_type(item, output_format)
        pages.append(
            MarkerPage(
                page=page_num,
                content=text,
                content_type=content_type,
                timing_ms=timing_ms,
            )
        )
    return pages


def _extract_pages_from_json_obj(data: Any) -> List[MarkerPage]:
    if isinstance(data, dict):
        output_format = data.get("output_format")

        if isinstance(data.get("pages"), list):
            return _pages_from_list(data["pages"], output_format)

        if isinstance(data.get("output"), list):
            return _pages_from_list(data["output"], output_format)

        if isinstance(data.get("output"), dict):
            nested = data["output"]
            nested_format = nested.get("output_format") or output_format
            if isinstance(nested.get("pages"), list):
                return _pages_from_list(nested["pages"], nested_format)

        if any(key in data for key in ("markdown", "md", "text", "content")):
            text, content_type = _extract_text_and_type(data, output_format)
            timing_ms = _extract_timing_ms(data)
            return [
                MarkerPage(
                    page=1,
                    content=text,
                    content_type=content_type,
                    timing_ms=timing_ms,
                )
            ]

        for key, value in data.items():
            if isinstance(key, str) and key.isdigit():
                try:
                    page_num = int(key)
                except ValueError:
                    continue
                text, content_type = _extract_text_and_type(value, output_format)
                pages = [
                    MarkerPage(
                        page=page_num,
                        content=text,
                        content_type=content_type,
                        timing_ms=_extract_timing_ms(value)
                        if isinstance(value, dict)
                        else None,
                    )
                ]
                return pages

    if isinstance(data, list):
        return _pages_from_list(data, None)

    if isinstance(data, str):
        content_type = "page_md" if "markdown" in data.lower() else "ocr_text"
        return [MarkerPage(page=1, content=data, content_type=content_type)]

    return []


def _read_json_file(path: Path) -> Optional[List[MarkerPage]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    pages = _extract_pages_from_json_obj(data)
    return pages or None


def _read_jsonl_file(path: Path) -> Optional[List[MarkerPage]]:
    pages: List[MarkerPage] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(data, dict):
                    page_num = _extract_page_num(data, index)
                    timing_ms = _extract_timing_ms(data)
                    output_format = data.get("output_format")
                    text, content_type = _extract_text_and_type(data, output_format)
                else:
                    page_num = index + 1
                    timing_ms = None
                    text, content_type = _extract_text_and_type(data, None)

                pages.append(
                    MarkerPage(
                        page=page_num,
                        content=text,
                        content_type=content_type,
                        timing_ms=timing_ms,
                    )
                )
    except OSError:
        return None

    return pages or None


def _collect_pages_from_files(output_dir: Path) -> List[MarkerPage]:
    text_files: List[Path] = []
    for ext in ("*.md", "*.markdown", "*.txt"):
        text_files.extend(output_dir.rglob(ext))

    with_numbers: List[Tuple[int, Path]] = []
    without_numbers: List[Path] = []

    for path in text_files:
        page_num = _parse_page_num_from_name(path)
        if page_num:
            with_numbers.append((page_num, path))
        else:
            without_numbers.append(path)

    pages: List[MarkerPage] = []
    if with_numbers:
        for page_num, path in sorted(with_numbers, key=lambda item: item[0]):
            content = path.read_text(encoding="utf-8", errors="ignore")
            content_type = (
                "page_md"
                if path.suffix.lower() in (".md", ".markdown")
                else "ocr_text"
            )
            pages.append(
                MarkerPage(page=page_num, content=content, content_type=content_type)
            )
        return pages

    if without_numbers:
        if len(without_numbers) == 1:
            path = without_numbers[0]
            content = path.read_text(encoding="utf-8", errors="ignore")
            content_type = (
                "page_md"
                if path.suffix.lower() in (".md", ".markdown")
                else "ocr_text"
            )
            if content_type == "page_md":
                split_pages = _split_marker_markdown_pages(content)
                if split_pages:
                    return [
                        MarkerPage(
                            page=page_num,
                            content=page_text,
                            content_type=content_type,
                        )
                        for page_num, page_text in split_pages
                    ]
            return [MarkerPage(page=1, content=content, content_type=content_type)]

        for index, path in enumerate(sorted(without_numbers)):
            content = path.read_text(encoding="utf-8", errors="ignore")
            content_type = (
                "page_md"
                if path.suffix.lower() in (".md", ".markdown")
                else "ocr_text"
            )
            pages.append(
                MarkerPage(page=index + 1, content=content, content_type=content_type)
            )
        return pages

    return []


def _collect_marker_pages(output_dir: Path) -> List[MarkerPage]:
    pages = _collect_pages_from_files(output_dir)
    if pages:
        return pages

    for path in sorted(output_dir.rglob("*.jsonl")):
        pages = _read_jsonl_file(path)
        if pages:
            return pages

    for path in sorted(output_dir.rglob("*.json")):
        pages = _read_json_file(path)
        if pages:
            return pages

    return []


def _looks_like_unknown_arg(stderr: str) -> bool:
    if not stderr:
        return False
    lowered = stderr.lower()
    return any(
        token in lowered
        for token in (
            "unrecognized argument",
            "unknown option",
            "no such option",
            "unknown argument",
        )
    )


def _build_marker_command(
    pdf_path: Path, output_dir: Path, page_range: Optional[str]
) -> Tuple[List[str], bool]:
    cmd_template = os.environ.get("BOOKMIND_MARKER_CMD") or os.environ.get(
        "MARKER_CMD"
    )
    if cmd_template:
        formatted = cmd_template.format(
            pdf=str(pdf_path), out=str(output_dir), pages=page_range or ""
        ).strip()
        return shlex.split(formatted), False

    marker_range = _to_zero_based_page_range(page_range) if page_range else None

    cmd = [
        "marker_single",
        str(pdf_path),
        "--output_dir",
        str(output_dir),
        "--output_format",
        "markdown",
        "--paginate_output",
    ]
    if marker_range:
        cmd.extend(["--page_range", marker_range])

    return cmd, True


def _ensure_marker_available(command: Sequence[str]) -> None:
    if not command:
        raise SystemExit("Marker command is empty.")

    if shutil.which(command[0]) is None:
        raise SystemExit(
            "Marker CLI not found. Install Marker locally and ensure the "
            "'marker_single' command is on PATH, or set BOOKMIND_MARKER_CMD to "
            "the correct command."
        )


def _run_marker_command(
    command: Sequence[str],
    allow_retry: bool,
    page_range: Optional[str],
    pdf_path: Path,
    output_dir: Path,
) -> None:
    _ensure_marker_available(command)

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if allow_retry and page_range and _looks_like_unknown_arg(stderr):
            fallback_cmd, _ = _build_marker_command(pdf_path, output_dir, None)
            subprocess.run(fallback_cmd, check=True)
            return

        message = (
            "Marker command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {exc.returncode}\n"
            f"Stdout: {exc.stdout}\n"
            f"Stderr: {stderr}"
        )
        raise SystemExit(message) from exc


def _normalize_marker_pages(
    pdf_path: Path, pages: List[MarkerPage], total_ms: int
) -> List[Dict[str, Any]]:
    if not pages:
        raise SystemExit("Marker produced no page outputs to normalize.")

    pages_sorted = sorted(pages, key=lambda item: item.page)
    fallback_ms = int(round(total_ms / max(1, len(pages_sorted))))
    timestamp_ms = int(time.time() * 1000)

    records: List[Dict[str, Any]] = []
    for page in pages_sorted:
        timing_ms = page.timing_ms if page.timing_ms is not None else fallback_ms
        records.append(
            {
                "engine": "marker",
                "type": page.content_type,
                "page": page.page,
                "timestamp_ms": timestamp_ms,
                "content": {"text": page.content},
                "meta": {
                    "pdf_page_start": page.page,
                    "pdf_page_end": page.page,
                    "pdf_path": str(pdf_path),
                },
                "timing_ms": timing_ms,
            }
        )

    return records


def run_marker_pdf(
    pdf_path: Path, output_dir: Path, page_range: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Run Marker locally and return normalized JSONL-ready records."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command, allow_retry = _build_marker_command(pdf_path, output_dir, page_range)

    start = time.perf_counter()
    _run_marker_command(command, allow_retry, page_range, pdf_path, output_dir)
    total_ms = int(round((time.perf_counter() - start) * 1000))

    pages = _collect_marker_pages(output_dir)
    return _normalize_marker_pages(pdf_path, pages, total_ms)


def _to_zero_based_page_range(page_range: str) -> str:
    parts: List[str] = []
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = [p.strip() for p in part.split("-", 1)]
            start = int(start_s)
            end = int(end_s)
            parts.append(f"{start - 1}-{end - 1}")
        else:
            value = int(part)
            parts.append(str(value - 1))
    return ",".join(parts)


def _split_marker_markdown_pages(text: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    current_page: Optional[int] = None
    buffer: List[str] = []

    for line in text.splitlines():
        match = _MARKER_PAGE_BREAK_RE.match(line.strip())
        if match:
            if current_page is not None:
                pages.append((current_page, "\n".join(buffer).strip()))
            current_page = int(match.group(1)) + 1
            buffer = []
            continue
        buffer.append(line)

    if current_page is not None:
        pages.append((current_page, "\n".join(buffer).strip()))

    return pages
