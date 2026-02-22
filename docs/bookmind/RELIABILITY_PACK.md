# Reliability Pack

`reliability-pack` combines retrieval-only quality signals with eval citation/verifier signals into one decision-ready artifact.

## What it contains

- Retrieval summary:
  - `hit_rate@k`
  - `mrr`
  - `avg_page_recall@k`
  - `stability` (when present in retrieval report)
- Generation quality summary:
  - `citations_resolvable_rate`
  - `avg_model_citations_per_answer`
  - `avg_context_citations_per_answer`
  - `avg_citations_per_answer` (backward-compatible alias for model average)
  - `verification_pass_rate`
  - `verification_status_counts`
- Decision fields:
  - `reliability_score` (0..100)
  - `meets_gate` (boolean)
  - `recommendation` (`PASS`, `INVESTIGATE_RETRIEVAL`, `INVESTIGATE_CITATIONS`, `INVESTIGATE_VERIFIER`)

## How to run

```bash
./tools/bookmind_bench/scripts/bench_python.sh tools/bookmind_bench/run.py reliability-pack \
  --retrieval_report /tmp/bookmind_retrieval_eval/mixed/retrieval_report.json \
  --eval_report /tmp/bookmind_eval_after_fix/report.json \
  --out /tmp/bookmind_reliability_pack
```

Output:
- `/tmp/bookmind_reliability_pack/reliability_report.json`

## Regression gate interpretation

Gate passes only if all are true:
- `Hit@20 >= 0.85`
- `MRR >= 0.35`
- `verification_pass_rate >= 0.80`
- `citations_resolvable_rate >= 0.90`

Recommendations:
- `PASS`: gate passed.
- `INVESTIGATE_RETRIEVAL`: retrieval metrics below threshold.
- `INVESTIGATE_CITATIONS`: citation resolvability below threshold.
- `INVESTIGATE_VERIFIER`: verifier pass-rate below threshold.

Citation metric semantics:
- `avg_context_citations_per_answer` comes from retrieval context rows included in prompts.
- `avg_model_citations_per_answer` comes from parsing citation markers in model answer text.
- `citations_resolvable_rate` is computed from model-citation parse+resolution (not context citation count).
- `verification_pass_rate` can be improved for gate-focused reruns by enabling eval citation policy:
  - `run.py eval --force_citations true`
  - Optional: `--citation_repair_retry true` and `--citation_min_count 1`
  - This adds strict citation instructions, an allowlist of retrieved citation IDs (top 30 by rank, `chunk_id` first then `stable_id`), and one repair retry when answers have too few markers.
  - Models must use only allowlisted IDs and copy them case-sensitively, which improves citation resolvability and reduces `FAIL_UNRESOLVABLE_CITATIONS`.
  - Default behavior is unchanged unless `--force_citations true` is set.

These thresholds are practical heuristics for scanned/OCR corpora, not hard scientific limits.

## Baselines and regression tracking

Use a frozen baseline folder under `docs/bookmind/baselines/<baseline_name>/` to track drift over time.

Suggested flow:
- Run a new E2E + Truth50 evaluation and produce `reliability_report.json` and `retrieval_report.json`.
- Compare the new `reliability_score`, `meets_gate`, and `recommendation` against the baseline.
- Compare retrieval metrics (`hit_rate@k`, `mrr`, `avg_page_recall@k`) against the baseline `retrieval_report.json`.
- If available, compare stability (`--repeat 5`) variance fields to detect nondeterministic retrieval regressions.
- Treat any drop below gate thresholds as a regression candidate and inspect per-query artifacts before changing thresholds.

## Compare to baseline

Use `compare-baseline` to generate an additive regression verdict from two
`reliability_report.json` files.

```bash
./tools/bookmind_bench/scripts/bench_python.sh tools/bookmind_bench/run.py compare-baseline \
  --baseline docs/bookmind/baselines/baseline_20260220_sample1-5/reliability_report.json \
  --current /tmp/bookmind_current/reliability_report.json \
  --out /tmp/bookmind_compare
```

Output:
- `/tmp/bookmind_compare/compare_report.json`

Verdicts:
- `PASS`: current report meets gate and no configured regression threshold is exceeded.
- `FAIL_GATE`: current `meets_gate` is false.
- `FAIL_REGRESSION`: gate passes, but one or more metric drops exceed thresholds.

Default regression thresholds:
- `reliability_score` drop `> 5.0`
- `verification_pass_rate` drop `> 0.05`
- `hit_rate@k` drop `> 0.05`
- `mrr` drop `> 0.05`

All thresholds are configurable via:
- `--max_reliability_score_drop`
- `--max_verification_pass_rate_drop`
- `--max_hit_rate_drop`
- `--max_mrr_drop`
