"""Bookmind offline benchmark CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from render_pdf import parse_pages, render_pdf_pages


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

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
