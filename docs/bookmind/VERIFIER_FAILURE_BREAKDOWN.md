# Verifier Failure Breakdown (Baseline `20260220_sample1-5`)

## Scope and source
- Baseline artifact: `docs/bookmind/baselines/baseline_20260220_sample1-5/reliability_report.json`
- Eval run dir used by baseline: `/tmp/bookmind_e2e_baseline_20260220_sample1-5`
- Eval report: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/eval/report.json`
- Query categories: `tools/bookmind_bench/samples/queries_scanned_tech_truth50.json`
- Note: `docs/bookmind/baselines/baseline_20260220_sample1-5/E2E_SUMMARY.md` has blank output paths; run dir was inferred from baseline metadata (`eval_report_path`).

## Executive signal
- `verification_pass_rate`: `0.0%` (`0/50`)
- All failures are `FAIL_NO_CITATIONS` despite `avg_num_citations=8.0` and `citations_resolvable_rate=1.0`.
- Decision: treat this as a verifier/citation-marker contract failure, not a retrieval failure.

## Verification status counts and rates

| verification_status | count | rate |
|---|---:|---:|
| PASS | 0 | 0.0% |
| FAIL_NO_CITATIONS | 50 | 100.0% |
| FAIL_UNRESOLVABLE_CITATIONS | 0 | 0.0% |
| FAIL_UNSUPPORTED_CLAIMS | 0 | 0.0% |

## `citations_resolvable` counts and rates

| citations_resolvable | count | rate |
|---|---:|---:|
| true | 50 | 100.0% |
| false | 0 | 0.0% |

## Breakdown by query category

| category | queries | PASS | FAIL | fail_rate |
|---|---:|---:|---:|---:|
| text | 19 | 0 | 19 | 100.0% |
| figure_caption | 11 | 0 | 11 | 100.0% |
| table | 11 | 0 | 11 | 100.0% |
| multi_page | 7 | 0 | 7 | 100.0% |
| not_in_book | 2 | 0 | 2 | 100.0% |

## Top 10 failing queries (by highest total query latency)

| id | query | category | verification_status | reason | citations_total/resolved | first_cited_rank |
|---|---|---|---|---|---|---|
| q42_tas_computation_pot | Which potentiometer is intended for TAS computation? | text | FAIL_NO_CITATIONS | Model answer has no citation markers. | 0/0 | null |
| q9_compass_components | Which components are labeled in the standby magnetic compass figure? | figure_caption | FAIL_NO_CITATIONS | Model answer has no citation markers. | 0/0 | null |
| q48_all_figure_index | Which figure numbers and titles are present in this scanned excerpt? | multi_page | FAIL_NO_CITATIONS | Model answer has no citation markers. | 0/0 | null |
| q44_tas_repeater_set | Which repeater potentiometers are listed for TAS? | table | FAIL_NO_CITATIONS | Model answer has no citation markers. | 0/0 | null |
| q24_pressure_difference_p | How is pressure p defined for the pitot-static aneroid? | text | FAIL_NO_CITATIONS | Model answer has no citation markers. | 0/0 | null |
| q38_r3_r4_bridge | Where is resistor bridge R3-R4 used in the signal path? | text | FAIL_NO_CITATIONS | Model answer has no citation markers. | 0/0 | null |
| q26_lighting_terminals | Which receptacle terminals supply lighting power? | text | FAIL_NO_CITATIONS | Model answer has no citation markers. | 0/0 | null |
| q33_tas_test_value | What computed TAS value is used for channel testing? | table | FAIL_NO_CITATIONS | Model answer has no citation markers. | 0/0 | null |
| q28_display_pointer_drive | What drives the display pointers according to the indicator description? | text | FAIL_NO_CITATIONS | Model answer has no citation markers. | 0/0 | null |
| q46_compass_vs_horizon | Compare standby magnetic compass content with standby horizon content. | multi_page | FAIL_NO_CITATIONS | Model answer has no citation markers. | 0/0 | null |

## Immediate implications
- The failure mode is uniform across all categories, indicating a formatting/parser contract issue in verifier citation extraction.
- Per-query artifacts show retrieved citation objects exist, but verifier fields stay `citations_total=0`, `citations_resolved=0`, `first_cited_rank=null` for all 50 queries.
