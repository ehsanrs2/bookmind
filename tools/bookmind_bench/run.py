"""Bookmind offline benchmark CLI."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from engines.marker_engine import run_marker_pdf
from engines.paddleocr_engine import (
    list_page_images,
    run_paddleocr,
    run_paddleocr_on_image,
)
from engines.vlm_hook import plan_vlm_jobs
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


def _resolve_paddleocr_cache_dir(arg_value: str | None) -> str | None:
    override = os.environ.get("BOOKMIND_PADDLEOCR_CACHE_DIR")
    if override:
        return override
    if arg_value:
        return arg_value
    default_dir = (
        Path(__file__).resolve().parent / "offline_bundle" / "models" / "paddleocr"
    )
    if default_dir.exists():
        return default_dir.as_posix()
    return None


def _cmd_paddleocr(args: argparse.Namespace) -> int:
    img_dir = Path(args.imgdir)
    out_dir = Path(args.out)

    if not img_dir.exists():
        raise SystemExit(f"Image directory not found: {img_dir}")

    images = list_page_images(str(img_dir))
    if not images:
        raise SystemExit(f"No page images found in {img_dir}")

    max_page = max(page for page, _ in images)
    pages = None
    if args.pages:
        pages = parse_pages(args.pages, max_page)
        if not pages:
            raise SystemExit("No pages selected. Check --pages range.")

    cache_dir = _resolve_paddleocr_cache_dir(args.cache_dir)
    records = run_paddleocr(
        img_dir=str(img_dir),
        pages=pages,
        lang=args.lang,
        use_gpu=args.use_gpu,
        cache_dir=cache_dir,
        debug_dir=args.debug_dir,
    )

    output_path = out_dir / "paddleocr" / "output.jsonl"
    write_jsonl(output_path, records)
    print(f"PaddleOCR wrote {len(records)} block(s) to {output_path}")
    return 0


def _cmd_plan_vlm(args: argparse.Namespace) -> int:
    paddleocr_jsonl = Path(args.paddleocr_jsonl)
    img_dir = Path(args.imgdir)
    out_path: Path
    if args.out:
        out_path = Path(args.out)
    else:
        if paddleocr_jsonl.name == "output.jsonl" and paddleocr_jsonl.parent.name == "paddleocr":
            out_path = paddleocr_jsonl.parent / "vlm_jobs.jsonl"
        else:
            raise SystemExit("Provide --out for plan-vlm when output path cannot be inferred.")

    if not paddleocr_jsonl.exists():
        raise SystemExit(f"PaddleOCR JSONL not found: {paddleocr_jsonl}")
    if not img_dir.exists():
        raise SystemExit(f"Image directory not found: {img_dir}")

    job_count = plan_vlm_jobs(
        records_jsonl_path=str(paddleocr_jsonl),
        imgdir=str(img_dir),
        out_jobs_path=str(out_path),
    )
    print(f"Planned {job_count} VLM job(s) to {out_path}")
    return 0


def _cmd_paddleocr_smoke(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    cache_dir = _resolve_paddleocr_cache_dir(args.cache_dir)
    records = run_paddleocr_on_image(
        image_path=str(image_path),
        lang=args.lang,
        use_gpu=args.use_gpu,
        cache_dir=cache_dir,
    )
    print(f"PaddleOCR smoke extracted {len(records)} block(s) from {image_path}")
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
        "paddleocr", help="Run PaddleOCR PP-Structure on rendered page images"
    )
    paddleocr.add_argument("--imgdir", required=True, help="Directory of PNG pages")
    paddleocr.add_argument("--out", required=True, help="Output directory for runs")
    paddleocr.add_argument(
        "--pages",
        required=False,
        help='Optional page ranges (1-based), e.g. "1-5,7,9-10"',
    )
    paddleocr.add_argument("--lang", default="en", help="OCR language (default en)")
    paddleocr.add_argument(
        "--use_gpu",
        default=True,
        type=lambda v: str(v).lower() in {"1", "true", "yes", "y"},
        help="Enable GPU (default true)",
    )
    paddleocr.add_argument(
        "--cache_dir",
        required=False,
        help="Override PaddleOCR cache dir (defaults to offline bundle if present)",
    )
    paddleocr.add_argument(
        "--debug_dir",
        required=False,
        help="Optional directory to write raw PP-Structure JSON per page",
    )
    paddleocr.set_defaults(func=_cmd_paddleocr)

    paddleocr_smoke = subparsers.add_parser(
        "paddleocr-smoke", help=argparse.SUPPRESS
    )
    paddleocr_smoke.add_argument("--image", required=True, help="Path to a PNG image")
    paddleocr_smoke.add_argument("--lang", default="en", help="OCR language")
    paddleocr_smoke.add_argument(
        "--use_gpu",
        default=True,
        type=lambda v: str(v).lower() in {"1", "true", "yes", "y"},
        help="Enable GPU (default true)",
    )
    paddleocr_smoke.add_argument(
        "--cache_dir",
        required=False,
        help="Override PaddleOCR cache dir (defaults to offline bundle if present)",
    )
    paddleocr_smoke.set_defaults(func=_cmd_paddleocr_smoke)

    plan_vlm = subparsers.add_parser(
        "plan-vlm", help="Plan VLM captioning jobs for figure regions"
    )
    plan_vlm.add_argument(
        "--paddleocr_jsonl",
        required=True,
        help="Path to PaddleOCR output.jsonl",
    )
    plan_vlm.add_argument("--imgdir", required=True, help="Directory of PNG pages")
    plan_vlm.add_argument(
        "--out",
        required=False,
        help="Output path for VLM job JSONL (defaults to <out>/paddleocr/vlm_jobs.jsonl)",
    )
    plan_vlm.set_defaults(func=_cmd_plan_vlm)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
