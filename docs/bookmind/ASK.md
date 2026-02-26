# BookMind Ask CLI

`tools/bookmind_bench/run.py ask` runs retrieval, context build, answer generation, citation parse/resolve, and additive verification in a single call.

## Usage

Basic:

```bash
./tools/bookmind_bench/scripts/bench_python.sh tools/bookmind_bench/run.py ask \
  --query "What does the TTAS channel do?" \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench
```

Write JSON artifact for UI:

```bash
./tools/bookmind_bench/scripts/bench_python.sh tools/bookmind_bench/run.py ask \
  --query "What does the TTAS channel do?" \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --top_k 20 \
  --content_types text,figure_caption \
  --backend ollama \
  --out /tmp/bookmind_ask/ask_result.json
```

Force citations + verification enforcement:

```bash
./tools/bookmind_bench/scripts/bench_python.sh tools/bookmind_bench/run.py ask \
  --query "What does the TTAS channel do?" \
  --force_citations true \
  --citation_min_count 1 \
  --citation_repair_retry auto \
  --enforce_verified true
```

## Key Args

- `--query` (required)
- `--qdrant_url`, `--collection`
- `--top_k` (default `20`)
- `--content_types` (default `text,figure_caption`)
- `--backend` (`ollama|vllm`) + backend-specific args already supported by `rag-preview`
- `--force_citations` (default `false`)
- `--citation_min_count` (default `1`)
- `--citation_repair_retry` (`auto|true|false`, default `auto`)
- `--enforce_verified` (default `false`)
- `--out` (optional path to write `ask_result.json`)

## Output JSON

`ask` always prints a stable JSON payload to stdout. If `--out` is set, the same payload is written to disk.

Fields:

- `query`
- `final_answer`
- `status`: `PASS | NOT_FOUND | FAIL`
- `verification_status`
- `verification_reason`
- `citations_total`
- `citations_resolved`
- `citations_resolvable`
- `cited_pages`
- `cited_ids`
- `first_cited_rank`
- `retrieved_rows`: top-k summary rows with `rank`, `score`, `page`, `stable_id`, `chunk_id`, `content_type`
- `config`: snapshot of `top_k`, `content_types`, `backend`, `force_citations`

## Status Rules

- If model answer is exactly `NOT_FOUND`, `status=NOT_FOUND`, `final_answer="NOT_FOUND"`, `verification_status=PASS`, and `verification_reason="NOT_FOUND"`.
- `NOT_FOUND` is an expected safe outcome and is treated as a verification pass.
- If verifier passes, `status=PASS`.
- Otherwise `status=FAIL`.
- With `--enforce_verified true`, non-`PASS` verification exits with non-zero code (`2`). JSON is still produced.
