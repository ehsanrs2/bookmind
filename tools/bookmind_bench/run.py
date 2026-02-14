"""Bookmind offline benchmark CLI."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from engines.marker_engine import run_marker_pdf
from engines.layout_engine import run_layoutparser
from engines.paddleocr_engine import (
    list_page_images,
    run_paddleocr,
    run_paddleocr_on_image,
)
from engines.vlm_caption_engine import (
    DEFAULT_BACKEND,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    run_vlm_caption_jobs,
    run_vlm_layout,
)
from engines.vlm_hook import plan_vlm_jobs
from bundle import bundle_run
from merge import merge_outputs
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
        verbose=args.verbose,
        force_ocr_fallback=args.force_ocr_fallback,
    )

    output_path = out_dir / "paddleocr" / "output.jsonl"
    write_jsonl(output_path, records)
    print(f"PaddleOCR wrote {len(records)} block(s) to {output_path}")
    return 0


def _resolve_layout_model_dir(arg_value: str | None) -> str:
    override = os.environ.get("BOOKMIND_LAYOUT_MODEL_DIR")
    if override:
        return override
    if arg_value:
        return arg_value
    return str(
        Path(__file__).resolve().parent
        / "offline_bundle"
        / "models"
        / "layoutparser_publaynet"
    )


def _cmd_layout(args: argparse.Namespace) -> int:
    img_dir = Path(args.imgdir)
    out_dir = Path(args.out)

    if not img_dir.exists():
        raise SystemExit(f"Image directory not found: {img_dir}")

    images = list_page_images(str(img_dir))
    if not images:
        raise SystemExit(f"No page images found in {img_dir}")

    model_dir = _resolve_layout_model_dir(args.model_dir)
    records = run_layoutparser(
        img_dir=str(img_dir),
        pages=None,
        use_gpu=args.use_gpu,
        model_dir=model_dir,
        score_thresh=args.score_thresh,
        max_regions_per_page=args.max_regions_per_page,
        out_dir=str(out_dir),
    )

    output_path = out_dir / "layout" / "output.jsonl"
    write_jsonl(output_path, records)
    print(f"LayoutParser wrote {len(records)} record(s) to {output_path}")
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


def _cmd_vlm(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    jobs_path = Path(args.jobs)
    paddleocr_path = Path(args.paddleocr_jsonl) if args.paddleocr_jsonl else None

    if not jobs_path.exists():
        raise SystemExit(f"VLM jobs JSONL not found: {jobs_path}")
    if paddleocr_path and not paddleocr_path.exists():
        raise SystemExit(f"PaddleOCR JSONL not found: {paddleocr_path}")

    backend = str(args.backend).strip().lower()
    if backend not in {"vllm", "ollama"}:
        raise SystemExit(f"Unsupported backend: {args.backend}")
    if backend == "ollama":
        if not str(args.ollama_url).strip():
            raise SystemExit("--ollama_url is required when --backend=ollama")
        if not str(args.ollama_model).strip():
            raise SystemExit("--ollama_model is required when --backend=ollama")
        selected_model = args.ollama_model
    else:
        selected_model = args.model
    print(f"[vlm] backend={backend} model={selected_model}")

    written = run_vlm_caption_jobs(
        jobs_jsonl=str(jobs_path),
        out_dir=str(out_dir),
        backend=backend,
        endpoint=args.endpoint,
        model=args.model,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        paddleocr_jsonl=str(paddleocr_path) if paddleocr_path else None,
        use_ocr_hints=args.use_ocr_hints,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    output_path = out_dir / "vlm" / "output.jsonl"
    print(f"VLM wrote {written} caption(s) to {output_path}")
    return 0


def _cmd_vlm_layout(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    layout_path = (
        Path(args.layout_json)
        if args.layout_json
        else out_dir / "layout" / "output.jsonl"
    )
    img_dir = Path(args.imgdir)

    if not layout_path.exists():
        raise SystemExit(f"Layout JSONL not found: {layout_path}")
    if not img_dir.exists():
        raise SystemExit(f"Image directory not found: {img_dir}")

    backend = str(args.backend).strip().lower()
    if backend not in {"vllm", "ollama"}:
        raise SystemExit(f"Unsupported backend: {args.backend}")
    if backend == "ollama":
        if not str(args.ollama_url).strip():
            raise SystemExit("--ollama_url is required when --backend=ollama")
        if not str(args.ollama_model).strip():
            raise SystemExit("--ollama_model is required when --backend=ollama")
        selected_model = args.ollama_model
    else:
        selected_model = args.model
    print(f"[vlm-layout] backend={backend} model={selected_model}")

    written = run_vlm_layout(
        layout_jsonl=str(layout_path),
        imgdir=str(img_dir),
        out_dir=str(out_dir),
        backend=backend,
        endpoint=args.endpoint,
        model=args.model,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        ocr_hint_max_chars=args.ocr_hint_max_chars,
        prompt_template=args.prompt_template,
        limit_image_size=args.limit_image_size,
    )
    output_path = out_dir / "vlm" / "output.jsonl"
    print(f"VLM layout wrote {written} caption(s) to {output_path}")
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)

    paddle_path = Path(args.paddle) if args.paddle else out_dir / "paddleocr" / "output.jsonl"
    if not paddle_path.exists():
        raise SystemExit(f"PaddleOCR JSONL not found: {paddle_path}")

    if args.vlm:
        vlm_path: Path | None = Path(args.vlm)
    else:
        inferred_vlm = out_dir / "vlm" / "output.jsonl"
        vlm_path = inferred_vlm if inferred_vlm.exists() else None

    if vlm_path is not None and not vlm_path.exists():
        raise SystemExit(f"VLM JSONL not found: {vlm_path}")

    output_path = out_dir / "ingest" / "records.jsonl"
    stats = merge_outputs(
        paddle_jsonl=paddle_path,
        vlm_jsonl=vlm_path,
        out_jsonl=output_path,
        min_text_chars=args.min_text_chars,
        bbox_tol=args.bbox_tol,
        iou_threshold=args.iou,
    )
    print(f"Merge wrote {stats.total_emitted} record(s) to {stats.output_path}")
    print(
        "By type: "
        f"text={stats.emitted_by_type.get('text', 0)} "
        f"table={stats.emitted_by_type.get('table', 0)} "
        f"figure_caption={stats.emitted_by_type.get('figure_caption', 0)}"
    )
    print(f"Figure captions matched: {stats.figure_captions_matched}")
    return 0


def _cmd_bundle(args: argparse.Namespace) -> int:
    result = bundle_run(
        run_dir=args.out,
        imgdir=args.imgdir,
        include_images=bool(args.include_images),
    )
    print(f"Bundle created at {result.bundle_dir}")
    print(
        f"Bundle records={result.records_count} copied_images={result.copied_images} "
        f"missing_images={result.missing_images}"
    )
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
    paddleocr.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose per-page logging for PaddleOCR",
    )
    paddleocr.add_argument(
        "--force_ocr_fallback",
        action="store_true",
        help="Always run OCR-only fallback (det+rec) per page",
    )
    paddleocr.set_defaults(func=_cmd_paddleocr)

    layout = subparsers.add_parser(
        "layout",
        help="Run region-aware LayoutParser + PaddleOCR on rendered page images",
    )
    layout.add_argument("--imgdir", required=True, help="Directory of PNG pages")
    layout.add_argument("--out", required=True, help="Output directory for runs")
    layout.add_argument(
        "--use_gpu",
        default=True,
        type=lambda v: str(v).lower() in {"1", "true", "yes", "y"},
        help="Enable GPU (default true)",
    )
    layout.add_argument(
        "--model_dir",
        required=False,
        help=(
            "Directory containing local LayoutParser PubLayNet config + weights "
            "(defaults to offline bundle)"
        ),
    )
    layout.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="Layout detection score threshold (default 0.5)",
    )
    layout.add_argument(
        "--max_regions_per_page",
        type=int,
        default=60,
        help="Maximum number of detected regions per page (default 60)",
    )
    layout.set_defaults(func=_cmd_layout)

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

    vlm = subparsers.add_parser(
        "vlm", help="Run VLM figure captioning jobs with a local endpoint"
    )
    vlm.add_argument("--jobs", required=True, help="Path to VLM jobs JSONL")
    vlm.add_argument("--out", required=True, help="Output directory for runs")
    vlm.add_argument(
        "--backend",
        choices=["vllm", "ollama"],
        default=DEFAULT_BACKEND,
        help="VLM backend (default vllm)",
    )
    vlm.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1",
        help="Local OpenAI-compatible VLM endpoint (default http://127.0.0.1:8000/v1)",
    )
    vlm.add_argument(
        "--model",
        default=os.environ.get("BOOKMIND_VLM_MODEL", "qwen3-vl"),
        help="Model name for the VLM endpoint (default env BOOKMIND_VLM_MODEL or qwen3-vl)",
    )
    vlm.add_argument(
        "--ollama_url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama URL for --backend ollama (default {DEFAULT_OLLAMA_URL})",
    )
    vlm.add_argument(
        "--ollama_model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model tag for --backend ollama (default {DEFAULT_OLLAMA_MODEL})",
    )
    vlm.add_argument(
        "--paddleocr_jsonl",
        required=False,
        help="Optional PaddleOCR JSONL for OCR hints",
    )
    vlm.add_argument(
        "--use_ocr_hints",
        default=True,
        type=lambda v: str(v).lower() in {"1", "true", "yes", "y"},
        help="Include OCR hints in the prompt (default true)",
    )
    vlm.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default 256)",
    )
    vlm.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default 0.2)",
    )
    vlm.set_defaults(func=_cmd_vlm)

    vlm_layout = subparsers.add_parser(
        "vlm-layout",
        help="Caption figure regions directly from LayoutParser output with local VLM",
    )
    vlm_layout.add_argument(
        "--layout_json",
        required=False,
        help="Path to layout output JSONL (default <out>/layout/output.jsonl)",
    )
    vlm_layout.add_argument("--imgdir", required=True, help="Directory of PNG pages")
    vlm_layout.add_argument("--out", required=True, help="Output directory for runs")
    vlm_layout.add_argument(
        "--backend",
        choices=["vllm", "ollama"],
        default=DEFAULT_BACKEND,
        help="VLM backend (default vllm)",
    )
    vlm_layout.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1",
        help="Local OpenAI-compatible VLM endpoint (default http://127.0.0.1:8000/v1)",
    )
    vlm_layout.add_argument(
        "--model",
        default=os.environ.get("BOOKMIND_VLM_MODEL", "qwen3-vl"),
        help="Model name for the VLM endpoint (default env BOOKMIND_VLM_MODEL or qwen3-vl)",
    )
    vlm_layout.add_argument(
        "--ollama_url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama URL for --backend ollama (default {DEFAULT_OLLAMA_URL})",
    )
    vlm_layout.add_argument(
        "--ollama_model",
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model tag for --backend ollama (default {DEFAULT_OLLAMA_MODEL})",
    )
    vlm_layout.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default 256)",
    )
    vlm_layout.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default 0.2)",
    )
    vlm_layout.add_argument(
        "--ocr_hint_max_chars",
        type=int,
        default=800,
        help="Max OCR hint characters in prompt (default 800)",
    )
    vlm_layout.add_argument(
        "--prompt_template",
        default="technical_diagram_v1",
        help="Prompt template id (default technical_diagram_v1)",
    )
    vlm_layout.add_argument(
        "--limit_image_size",
        type=int,
        required=False,
        help="Optional max crop side length in pixels (e.g. 384)",
    )
    vlm_layout.set_defaults(func=_cmd_vlm_layout)

    merge = subparsers.add_parser(
        "merge", help="Merge PaddleOCR + optional VLM outputs into ingest JSONL"
    )
    merge.add_argument(
        "--paddle",
        required=False,
        help="Path to PaddleOCR output JSONL (default <out>/paddleocr/output.jsonl)",
    )
    merge.add_argument(
        "--vlm",
        required=False,
        help="Optional path to VLM output JSONL (default <out>/vlm/output.jsonl if present)",
    )
    merge.add_argument("--out", required=True, help="Output run directory")
    merge.add_argument(
        "--min-text-chars",
        type=int,
        default=20,
        help="Skip short OCR text blocks below this length (default 20)",
    )
    merge.add_argument(
        "--bbox-tol",
        type=int,
        default=2,
        help="Per-coordinate bbox tolerance in pixels (default 2)",
    )
    merge.add_argument(
        "--iou",
        type=float,
        default=0.95,
        help="IoU threshold for fallback figure-caption matching (default 0.95)",
    )
    merge.set_defaults(func=_cmd_merge)

    bundle = subparsers.add_parser(
        "bundle", help="Create portable ingest bundle + quality report"
    )
    bundle.add_argument("--out", required=True, help="Run directory")
    bundle.add_argument(
        "--imgdir",
        required=False,
        help="Optional page images directory for resolving figure image references",
    )
    bundle.add_argument(
        "--include_images",
        default=False,
        type=lambda v: str(v).lower() in {"1", "true", "yes", "y"},
        help="Copy figure page/crop images into bundle/images (default false)",
    )
    bundle.set_defaults(func=_cmd_bundle)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
