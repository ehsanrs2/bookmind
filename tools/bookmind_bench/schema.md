# Bookmind Benchmark Schema

This scaffold defines a **normalized JSONL** format for offline extraction outputs. Each line is a single JSON object representing a page-level extraction result.

## Directory layout (recommended)
```
./tools/bookmind_bench/
  run.py
  render_pdf.py
  io.py
  schema.md

./bench_runs/<run_id>/
  input/
    <document>.pdf
  images/
    page_0001.png
    page_0002.png
  outputs/
    pages.jsonl
  metadata/
    run.json
```

## JSONL schema (page-level)
Each line in `pages.jsonl` should follow this normalized shape:

```json
{
  "run_id": "2026-02-01T120000Z_demo",
  "doc_id": "document.pdf",
  "pdf_path": "bench_runs/2026-02-01T120000Z_demo/input/document.pdf",
  "page_num": 1,
  "image_path": "bench_runs/2026-02-01T120000Z_demo/images/page_0001.png",
  "dpi": 300,
  "timestamp_ms": 1706788800000,
  "content": {
    "text": "...full page text...",
    "blocks": [
      {
        "id": "b1",
        "type": "text",
        "bbox": [x0, y0, x1, y1],
        "text": "...",
        "confidence": 0.99
      }
    ],
    "tables": [
      {
        "id": "t1",
        "bbox": [x0, y0, x1, y1],
        "rows": [["A", "B"], ["1", "2"]]
      }
    ]
  },
  "timings_ms": {
    "render": 123,
    "extract": 456
  },
  "errors": []
}
```

### Field notes
- `run_id`: Unique run identifier (timestamp + label works well).
- `doc_id`: Stable document identifier (filename or hash).
- `page_num`: 1-based page number.
- `bbox`: `[x0, y0, x1, y1]` in pixel space of the rendered image.
- `confidence`: Optional float 0..1.
- `errors`: List of error strings for page-level failures (empty on success).

This schema is intentionally permissive so different extraction backends can be compared offline without network access.

## Marker engine

Command example:
```
python tools/bookmind_bench/run.py marker --pdf path/to/input.pdf --out bench_runs --pages "1-5"
```

Output location:
```
<out>/marker/output.jsonl
```

Page filtering behavior:
- The runner passes `--pages`/page-range args to Marker when supported.
- If the local Marker CLI does not accept page ranges, it will process the full PDF
  and filter the normalized JSONL records by the requested page numbers.

## PaddleOCR PP-Structure engine

Command example:
```
python tools/bookmind_bench/run.py paddleocr --imgdir bench_runs/images --out bench_runs --pages "1-5"
```

Output location:
```
<out>/paddleocr/output.jsonl
```

Output schema (multi-block JSONL):
- Each line is a region-level record with `engine="paddleocr"`.
- `page` is 1-based, `bbox` is `[x0, y0, x1, y1]` in pixel space.
- `type` values:
  - `ocr_text`: text-like regions, `text` contains region-specific OCR.
  - `table_md`: table regions, `text` contains Markdown (best-effort).
  - `layout_block`: non-text regions (e.g. figures), `text` is empty or brief.
- `meta.block_type` is normalized to one of:
  - `text`, `title`, `list`, `table`, `figure`, `equation`, `unknown`.
- `meta.confidence` is optional.
- `timing_ms` is the per-page processing time (same for all records in that page).

Figure handling:
- Figures are emitted as `layout_block` with `meta.block_type="figure"`.
- Text labels inside figures should appear as separate `ocr_text` blocks if detected by PP-Structure.
- Fallback for scanned manuals: if PP-Structure returns no usable text or table regions, an OCR-only pass runs to
  emit multiple `ocr_text` blocks. The original figure block is still emitted so VLM cropping/captioning can
  proceed. A secondary heuristic triggers fallback when PP-Structure returns a single figure region covering
  >= 80% of the page.
- Knobs (constants in `paddleocr_engine.py`): `FULLPAGE_FIGURE_AREA_RATIO` (default 0.80),
  `MIN_TEXT_CHARS` (default 2), `MAX_TEXT_BLOCKS_PER_PAGE` (default 500).
- CLI flags: `--verbose` prints per-page diagnostics, `--force_ocr_fallback` always runs OCR-only fallback.

Debug raw output:
- If `--debug_dir` is set, raw PP-Structure JSON is written to:
  `<debug_dir>/paddleocr_raw/page_<N>.json`.
- OCR fallback raw output (when triggered) is written to:
  `<debug_dir>/paddleocr_ocr_raw/page_<N>.json`.

Offline cache behavior:
- Runtime downloads are disabled. Ensure `tools/bookmind_bench/offline_bundle/models/paddleocr`
  is populated via `scripts/prefetch_online.sh`.
- You can override the cache directory with `BOOKMIND_PADDLEOCR_CACHE_DIR` or `--cache_dir`.

## LayoutParser region-aware engine (PubLayNet + PaddleOCR crops)

Command example:
```
python tools/bookmind_bench/run.py layout --imgdir bench_runs/images --out bench_runs
```

Output location:
```
<out>/layout/output.jsonl
```

Optional debug artifacts:
```
<out>/layout/crops/page_0001_region_001.png
<out>/layout/layout_debug/page_0001.json
```

Output schema (multi-record JSONL):
- `engine="layoutparser"` for all records.
- `page` is 1-based.
- `type` values:
  - `layout_block`: one record per detected region (`text`, `table`, `figure`), with region bbox and detection metadata.
  - `ocr_text`: OCR lines from cropped `text`/`table` regions, mapped back into page coordinates.
  - `table_md`: reserved/optional (not emitted by default in this path).
- `meta` always includes `pdf_page_start` and `pdf_page_end`.
- `layout_block` records include:
  - `meta.block_type` (`text`, `table`, `figure`)
  - `meta.confidence` (optional detector score)
  - `meta.region_id` (`page_0001_region_001` style)
- `ocr_text` records include:
  - `meta.block_type` (parent region type)
  - `meta.trace.region_id` linking each line back to its region.

Offline model behavior:
- The layout model path defaults to:
  - `tools/bookmind_bench/offline_bundle/models/layoutparser_publaynet`
- Override with `--model_dir` or `BOOKMIND_LAYOUT_MODEL_DIR`.
- Runtime avoids network; provide local Detectron2 config (`.yaml`) + weights (`.pth/.pkl`) in `model_dir`.

## VLM planning hook (figure captioning)

Command example:
```
python tools/bookmind_bench/run.py plan-vlm --paddleocr_jsonl bench_runs/paddleocr/output.jsonl --imgdir bench_runs/images
```

Output location (default when PaddleOCR output path is standard):
```
<out>/paddleocr/vlm_jobs.jsonl
```

Job JSONL schema:
```json
{
  "page": 1,
  "bbox": [x0, y0, x1, y1],
  "image_path": "bench_runs/images/page_0001.png",
  "suggested_crop_path": "bench_runs/paddleocr/vlm_crops/page_0001_block_001.png",
  "prompt_template": "technical_diagram_v1"
}
```

## VLM caption runner (figure crops)

Command example:
```
python tools/bookmind_bench/run.py vlm --jobs bench_runs/paddleocr/vlm_jobs.jsonl --out bench_runs --endpoint http://127.0.0.1:8000/v1 --model qwen3-vl
```

Optional OCR hints:
```
python tools/bookmind_bench/run.py vlm --jobs bench_runs/paddleocr/vlm_jobs.jsonl --out bench_runs --paddleocr_jsonl bench_runs/paddleocr/output.jsonl --use_ocr_hints true
```

Output location:
```
<out>/vlm/output.jsonl
```

Requirements:
- Start a local OpenAI-compatible VLM endpoint (vLLM) before running `vlm`.
- The endpoint must be reachable at `--endpoint` (default `http://127.0.0.1:8000/v1`).
- No remote downloads are performed by the runner.

Output schema (figure caption JSONL):
```json
{
  "engine": "vlm",
  "page": 1,
  "type": "figure_caption",
  "text": "Concise technical caption describing components and flow...",
  "bbox": [x0, y0, x1, y1],
  "timing_ms": 1234,
  "meta": {
    "pdf_page_start": 1,
    "pdf_page_end": 1,
    "prompt_template": "technical_diagram_v1",
    "model": "qwen3-vl",
    "job_index": 1,
    "used_ocr_hints": true
  }
}
```

OCR hints behavior:
- If `--paddleocr_jsonl` is provided and `--use_ocr_hints true`, the runner gathers `ocr_text` blocks
  whose bbox intersects the figure bbox (IoU > 0.01).
- Hints are truncated to keep prompts compact (default 800 chars, 30 items).

## Merge output (ingest_record)

This is a bench-only normalization step that prepares artifacts for later
Bookmind/OpenWebUI ingestion integration.

Command example:
```
python tools/bookmind_bench/run.py merge --out <run_dir>
```

Input defaults:
- PaddleOCR: `<run_dir>/paddleocr/output.jsonl`
- VLM (optional): `<run_dir>/vlm/output.jsonl` when present

Output location:
```
<run_dir>/ingest/records.jsonl
```

Output schema (`ingest_record`, one JSON object per line):
```json
{
  "stable_id": "8f745148f0f56061",
  "page": 1,
  "content_type": "text",
  "text": "Normalized text content",
  "bbox": [12, 34, 420, 460],
  "meta": {
    "pdf_page_start": 1,
    "pdf_page_end": 1,
    "source_engines": ["paddleocr"],
    "block_type": "text",
    "confidence": 0.91,
    "trace": {
      "paddleocr_line_ids": [14],
      "vlm_line_id": null
    }
  }
}
```

Field notes:
- `stable_id`: deterministic ID from `sha1(f"{page}|{type}|{bbox_norm}|{text_norm}")[:16]`.
- `content_type`: one of `text`, `table`, `figure_caption`.
- `bbox`: rounded integer bbox `[x0, y0, x1, y1]` or `null`.
- `source_engines`: `["paddleocr"]` for text/tables and `["paddleocr","vlm"]` for matched figure captions.
- Figure-caption matching is page-aware and uses exact bbox first, then coordinate tolerance (`--bbox-tol`), then IoU fallback (`--iou`).
