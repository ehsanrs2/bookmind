# BookMind UI Integration

Use either:
- `ask_json.sh` for CLI wrapper mode, or
- the local HTTP API (`run.py serve`) for API mode.

Both modes return the same stable ask JSON payload schema.

API mode details and examples: `docs/bookmind/API.md`.

## Wrapper Script

```bash
./tools/bookmind_bench/scripts/ask_json.sh \
  --query "What does the TTAS channel do?" \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --backend ollama \
  --ollama_url http://127.0.0.1:11434 \
  --ollama_model qwen3-vl:latest \
  --ollama_api generate
```

Optional citation/verification gating:

```bash
./tools/bookmind_bench/scripts/ask_json.sh \
  --query "What does the TTAS channel do?" \
  --force_citations true \
  --enforce_verified true
```

Notes:
- If `--out` is provided, JSON is written there and also printed by `run.py ask`.
- If `--out` is not provided, the wrapper writes to a temp file and prints that JSON to stdout.

## Expected JSON Fields

Top-level fields emitted by `run.py ask`:

- `query`: input question string.
- `final_answer`: model answer text, or `NOT_FOUND`.
- `status`: `PASS | NOT_FOUND | FAIL`.
- `verification_status`: verifier code (for example `PASS`, citation failures, or other failure code).
- `verification_reason`: short verifier reason or generation error text.
- `citations_total`: total citation markers parsed from the model answer.
- `citations_resolved`: number of parsed citations that resolved to retrieved rows.
- `citations_resolvable`: boolean indicating whether all required citations were resolvable.
- `cited_pages`: sorted list of cited page numbers.
- `cited_ids`: list of resolved stable IDs.
- `first_cited_rank`: rank of first cited retrieved row (or null if none).
- `retrieved_rows`: array of retrieval summaries.
- `config`: effective ask config snapshot.

`retrieved_rows[*]` fields:

- `rank`
- `score`
- `page`
- `stable_id`
- `chunk_id`
- `content_type`

`config` fields:

- `top_k`
- `content_types`
- `backend`
- `force_citations`

## Product Regression Gate

Run a small product gate quickly with the stored baseline checks:

```bash
python tools/bookmind_bench/run.py product-check \
  --checks docs/bookmind/baselines/product_baseline_20260226/product_checks.json \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --backend ollama \
  --ollama_url http://127.0.0.1:11434 \
  --ollama_model qwen3-vl:latest \
  --out /tmp/bookmind_product_gate_current
```

Outputs:
- Per-check JSON: `/tmp/bookmind_product_gate_current/<check_id>.json`
- Summary: `/tmp/bookmind_product_gate_current/report.json`

Exit code behavior:
- `0`: all required PASS/NOT_FOUND checks satisfied and safety invariants held.
- `2`: one or more required expectations or safety invariants failed.

Compare current run against a stored baseline report:

```bash
python tools/bookmind_bench/run.py compare-product-baseline \
  --baseline_report docs/bookmind/baselines/product_baseline_20260226/report.json \
  --current_report /tmp/bookmind_product_gate_current/report.json \
  --out /tmp/bookmind_product_gate_compare
```

Optional threshold:
- `--max_fail_increase <N>` fails comparison when total failed checks increase by more than `N`.
