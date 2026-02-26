# BookMind E2E Run

Use the E2E orchestrator to run extraction, ingest, optional evaluation, and reliability packaging in one command.

For product-facing single-query inference output, see `docs/bookmind/ASK.md`.

## Example (`--pdf` Mode)

```bash
./tools/bookmind_bench/scripts/e2e_bookmind.sh \
  --pdf /path/to/book.pdf \
  --out /tmp/bookmind_e2e_run \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --backend ollama \
  --ollama_url http://127.0.0.1:11434 \
  --ollama_model qwen3-vl:latest \
  --ollama_api generate \
  --queries tools/bookmind_bench/samples/queries_scanned_tech_truth50.json
```

## Example (`--from_bundle` Mode)

Use this mode when a bundle already exists and you want to start directly from ingest.

```bash
./tools/bookmind_bench/scripts/e2e_bookmind.sh \
  --from_bundle /tmp/bookmind_e2e_run/bundle \
  --out /tmp/bookmind_e2e_rerun \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --backend ollama \
  --ollama_url http://127.0.0.1:11434 \
  --ollama_model qwen3-vl:latest \
  --ollama_api generate \
  --queries tools/bookmind_bench/samples/queries_scanned_tech_truth50.json
```

Behavior notes:
- `--from_bundle` skips extraction, merge, and bundle creation.
- Pipeline starts at `qdrant-ingest` using the provided bundle path.
- In `--pdf` mode, extractor steps that are unavailable are skipped with warnings when optional.

## What It Produces

- `out/images/`: rendered page PNGs.
- `out/marker/output.jsonl`: marker extraction output.
- `out/paddleocr/output.jsonl`: OCR output used by merge.
- `out/layout/output.jsonl`: layout extraction output (when model is available).
- `out/vlm/output.jsonl`: VLM layout captions (when layout runs and backend is configured).
- `out/ingest/records.jsonl`: merged ingest records.
- `out/bundle/`: portable bundle for ingestion.
- `out/retrieval_eval/retrieval_report.json`: retrieval metrics (when `--queries` is provided).
- `out/eval/report.json`: answer-generation eval report (when `--queries` is provided).
- `out/reliability/reliability_report.json`: combined gate artifact (when `--queries` is provided).
- `out/E2E_SUMMARY.md`: final run summary with step status and artifact paths.
- `out/logs/*.log`: per-step logs.

For reliability gating, check:

- `out/reliability/reliability_report.json`

## Citation-Policy Mode for Eval Gating

To reduce `FAIL_NO_CITATIONS` during benchmark reruns, run eval with force-citation policy:

```bash
./tools/bookmind_bench/scripts/bench_python.sh tools/bookmind_bench/run.py eval \
  --queries tools/bookmind_bench/samples/queries_scanned_tech_truth50.json \
  --out /tmp/bookmind_eval \
  --force_citations true \
  --citation_min_count 1
```

Optional knobs:
- `--citation_repair_retry true|false`
- `--citation_min_count <int>`

Behavior:
- With `--force_citations true`, prompts require bracketed citation markers and allow `NOT_FOUND` when context is insufficient.
- In force mode, prompts also include an allowlist of citation IDs extracted from retrieved context rows (top 30 by rank, prefer `chunk_id`, fallback `stable_id`).
- Citations are case-sensitive in this allowlist; models are instructed to copy IDs exactly and use only listed IDs.
- This reduces `FAIL_UNRESOLVABLE_CITATIONS` by preventing invented/non-retrieved citation IDs.
- If enabled, one repair retry rewrites answers with citations when initial output has too few markers.
- In force mode repair, unsupported sentences may be removed when they cannot be cited with allowed IDs; if no supported answer remains, output is `NOT_FOUND`.
- Default behavior remains unchanged when `--force_citations false` (default).
