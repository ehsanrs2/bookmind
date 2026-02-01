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
