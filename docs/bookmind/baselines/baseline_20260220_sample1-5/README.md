# Baseline: baseline_20260220_sample1-5

- Date: 2026-02-20
- PDF: `sample/sample1-5.pdf`
- PDF SHA256: `4a4cfbe27aaeef2c8fc64b5ebbc408d0a5174dce2b239316ca0954d67db75c9d`
- Qdrant collection: `bookmind_bench`
- Runner: `tools/bookmind_bench/scripts/e2e_bookmind.sh`
- Queries: `tools/bookmind_bench/samples/queries_scanned_tech_truth50.json`
- Backend/model: `ollama` / `qwen3-vl:latest` (`ollama_api=generate`)
- Output run dir: `/tmp/bookmind_e2e_baseline_20260220_sample1-5`

## Key metrics (from `reliability_report.json`)

- `reliability_score`: `61.356`
- `meets_gate`: `false`
- `recommendation`: `INVESTIGATE_VERIFIER`

- Retrieval summary:
- `hit_rate@k`: `0.96`
- `mrr`: `0.87804`
- `avg_page_recall@k`: `0.96`
- `stability.avg_jaccard_topk`: `1.0`
- `stability.avg_jaccard_top5`: `1.0`

- Generation quality summary:
- `citations_resolvable_rate`: `1.0`
- `avg_citations_per_answer`: `8.0`
- `verification_pass_rate`: `0.0`
- `verification_status_counts`: `{"FAIL_NO_CITATIONS": 50}`

## Notes

- This baseline uses the mixed retrieval scope (`text,figure_caption`) with a separate stability pass (`--repeat 5`) saved as `retrieval_report.json` in this baseline folder.
