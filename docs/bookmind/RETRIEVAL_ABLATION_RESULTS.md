# Retrieval Ablation Results

Date: 2026-02-20
Truth set: `tools/bookmind_bench/samples/queries_scanned_tech_truth25.json` (currently 8 annotated queries)
Top-k: 20
Collection: `bookmind_bench`

## Summary metrics

| Setup | hit_rate@k | avg_page_recall@k | mrr |
|---|---:|---:|---:|
| text-only (`content_types=text`) | 1.0000 | 0.8750 | 0.6771 |
| caption-only (`content_types=figure_caption`) | 0.8750 | 0.6458 | 0.7292 |
| mixed (`content_types=text,figure_caption`) | 1.0000 | 1.0000 | 0.7917 |

## Notable findings

- Mixed is best overall: perfect `hit_rate@k` and `avg_page_recall@k`, with the highest `mrr`.
- Text-only has full hit-rate but misses some truth coverage on multi-page queries (`q3_parts_list`, `q8_calibration_figure` at recall 0.5 each).
- Caption-only has strong early ranking on figure-heavy queries (higher MRR than text-only), but lower completeness and one miss (`q5_table_values`, recall 0.0).

## By-content-type breakdown (from reports)

- Text-only report:
  - `text`: `retrieved_count=160`, `retrieved_fraction=1.0`, `truth_page_row_hits=101`
- Caption-only report:
  - `figure_caption`: `retrieved_count=40`, `retrieved_fraction=1.0`, `truth_page_row_hits=16`
- Mixed report:
  - `text`: `retrieved_count=132`, `retrieved_fraction=0.825`, `truth_page_row_hits=82`
  - `figure_caption`: `retrieved_count=28`, `retrieved_fraction=0.175`, `truth_page_row_hits=15`

Interpretation: in mixed mode, text rows dominate retrieval volume while figure captions still contribute meaningful truth-page hits, yielding the best completeness.

## Artifacts

- `/tmp/bookmind_retrieval_eval/text_only/retrieval_report.json`
- `/tmp/bookmind_retrieval_eval/caption_only/retrieval_report.json`
- `/tmp/bookmind_retrieval_eval/mixed/retrieval_report.json`

## Runtime note

Running with system `python` failed due to missing `qdrant_client`; ablations were run successfully with `./.venv_bookmind_bench_qdrant/bin/python` and no code patch was required.

For reproducible bench runs, use the launcher:
- `./tools/bookmind_bench/scripts/bench_python.sh tools/bookmind_bench/run.py ...`
- Environment setup reference: `docs/bookmind/BENCH_ENV.md`

## Stability

Repeat-run configuration:
- setup: mixed (`content_types=text,figure_caption`)
- repeats: `5`
- seed: `1337`
- top_k: `20`
- report: `/tmp/bookmind_retrieval_eval/mixed_repeat5/retrieval_report.json`

Observed stability metrics:
- `stability.avg_jaccard_topk = 1.0`
- `stability.avg_jaccard_top5 = 1.0`

Interpretation (heuristics, not hard rules):
- `avg_jaccard_top5 >= 0.8` is strong stability.
- `avg_jaccard_topk >= 0.6` is acceptable for scanned/OCR corpora.

Based on these heuristics, this run shows very strong retrieval stability at both top-5 and top-k.

## Verifier

An additive verifier stage is now enabled in bench flows to reduce hallucinations without changing retrieval logic or generation prompts.

- It checks:
  - citation presence,
  - citation resolvability against retrieved rows,
  - conservative grounding overlap between answer terms and retrieved text.
- In `rag-preview`, failures surface as a verification block and can be enforced with `--enforce_verified true`.
- In `eval`, reports now include per-query verification status/reason and aggregate `verification_pass_rate`.
