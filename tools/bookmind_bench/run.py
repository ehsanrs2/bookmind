"""Bookmind offline benchmark CLI."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from engines.marker_engine import run_marker_pdf
from render_pdf import parse_pages, render_pdf_pages


def _load_local_io() -> object:
    io_path = Path(__file__).resolve().parent / "io.py"
    spec = importlib.util.spec_from_file_location("bookmind_bench_io", io_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load local io module at {io_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_io = _load_local_io()
write_jsonl = _io.write_jsonl


def _cmd_render(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    # Determine max pages for parsing
    import fitz  # PyMuPDF

    with fitz.open(pdf_path) as doc:
        max_page = doc.page_count

    pages = parse_pages(args.pages, max_page)
    if not pages:
        raise SystemExit("No pages selected. Check --pages range.")

    result = render_pdf_pages(
        pdf_path=pdf_path,
        output_dir=out_dir,
        pages=pages,
        dpi=args.dpi,
    )

    print(f"Rendered {len(result.image_paths)} page(s) to {result.output_dir}")
    return 0


def _cmd_marker(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    import fitz  # PyMuPDF

    with fitz.open(pdf_path) as doc:
        max_page = doc.page_count

    pages = None
    if args.pages:
        pages = parse_pages(args.pages, max_page)
        if not pages:
            raise SystemExit("No pages selected. Check --pages range.")

    marker_out_dir = out_dir / "marker"
    records = run_marker_pdf(
        pdf_path=pdf_path,
        output_dir=marker_out_dir,
        page_range=args.pages if args.pages else None,
    )

    if pages is not None:
        page_set = set(pages)
        records = [record for record in records if record.get("page") in page_set]
        if not records:
            raise SystemExit("Marker produced no records for the selected pages.")

    output_path = marker_out_dir / "output.jsonl"
    write_jsonl(output_path, records)
    print(f"Marker wrote {len(records)} page(s) to {output_path}")
    return 0


def _cmd_paddleocr(_args: argparse.Namespace) -> int:
    raise SystemExit(
        "PaddleOCR engine integration is not implemented yet. "
        "Use this command as a placeholder for offline verification."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bookmind-bench",
        description="Offline benchmark harness for Bookmind extraction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="Render PDF pages to PNG images")
    render.add_argument("--pdf", required=True, help="Path to input PDF")
    render.add_argument("--out", required=True, help="Output directory for images")
    render.add_argument(
        "--pages",
        required=True,
        help='Page ranges (1-based), e.g. "1-5,7,9-10"',
    )
    render.add_argument("--dpi", type=int, default=300, help="Render DPI (default 300)")
    render.set_defaults(func=_cmd_render)

    marker = subparsers.add_parser("marker", help="Run Marker on a PDF locally")
    marker.add_argument("--pdf", required=True, help="Path to input PDF")
    marker.add_argument("--out", required=True, help="Output directory for runs")
    marker.add_argument(
        "--pages",
        required=False,
        help='Optional page ranges (1-based), e.g. "1-5,7,9-10"',
    )
    marker.set_defaults(func=_cmd_marker)

    paddleocr = subparsers.add_parser(
        "paddleocr", help="Run PaddleOCR/PP-Structure (placeholder)"
    )
    paddleocr.set_defaults(func=_cmd_paddleocr)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
