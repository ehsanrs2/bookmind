# Verifier Failure Breakdown (Post-fix Rerun)

Inputs used:
- `/tmp/bookmind_e2e_rerun_20260220_sample1-5/eval/report.json`
- `tools/bookmind_bench/samples/queries_scanned_tech_truth50.json`
- `/tmp/bookmind_reliability_pack_rerun/reliability_report.json`

Sample size: **50 queries**
Current verification pass rate: **0.42 (21/50)**
Gate target: **>= 0.80 (>= 40/50)**
Additional passes required to meet gate: **19**

## 1) verification_status counts/rates

| verification_status | count | rate |
|---|---:|---:|
| PASS | 21 | 42.0% |
| FAIL_NO_CITATIONS | 24 | 48.0% |
| FAIL_UNRESOLVABLE_CITATIONS | 4 | 8.0% |
| FAIL_UNSUPPORTED_CLAIMS | 1 | 2.0% |

Observation: failures are dominated by `FAIL_NO_CITATIONS` (24/29 fails, 82.8% of fails).

## 2) Breakdown by category

| category | total | pass | pass rate | fail_no_citations | fail_unresolvable | fail_unsupported |
|---|---:|---:|---:|---:|---:|---:|
| text | 19 | 11 | 57.9% | 8 | 0 | 0 |
| figure_caption | 11 | 7 | 63.6% | 2 | 1 | 1 |
| table | 11 | 1 | 9.1% | 8 | 2 | 0 |
| multi_page | 7 | 2 | 28.6% | 4 | 1 | 0 |
| not_in_book | 2 | 0 | 0.0% | 2 | 0 | 0 |

Key concentration: `table` + `multi_page` account for 13/29 failures (44.8%), mostly `FAIL_NO_CITATIONS`.

## 3) FAIL_UNSUPPORTED_CLAIMS overlap distribution

Overlap stats were available in `verification_reason`.

`FAIL_UNSUPPORTED_CLAIMS` cases (n=1):
- `q17_aoa_flow_chain`: matched **1/10** key terms (10% overlap), threshold required `>=2`.

Distribution summary:
- `1/10`: 1 case (100%)

## 4) FAIL_UNRESOLVABLE_CITATIONS top queries (citations_total>0 and resolvable=false)

Only 4 queries match this condition in the rerun (requested top 10 shown up to available size):

| rank | query id | citations_total | citations_resolved | resolvable | query |
|---:|---|---:|---:|---|---|
| 1 | `q6_block_diagram` | 4 | 0 | false | Explain the block diagram and how the modules connect. |
| 2 | `q1_title_scope` | 3 | 2 | false | What is the title or main subject of this document? |
| 3 | `q33_tas_test_value` | 3 | 2 | false | What computed TAS value is used for channel testing? |
| 4 | `q25_thermal_compensation_range` | 1 | 0 | false | What thermal compensation range is specified for the indicator device? |

## What must change to reach verification_pass_rate >= 0.80?

### Quantified calibration impact

- Current: **21/50 pass (0.42)**.
- Required: **40/50 pass (0.80)**.
- Minimum net gain: **+19 passes**.

Scenario math:
- If you only reduce `FAIL_UNSUPPORTED_CLAIMS` by lowering overlap threshold (`>=2` to `>=1`), gain is **+1** at most: **22/50 = 0.44** (insufficient).
- If you exclude `not_in_book` from denominator: pass rate becomes **21/48 = 0.4375** (small change, still far from gate).
- If `FAIL_NO_CITATIONS` conversions to PASS are:
  - 75% conversion (18/24): **39/50 = 0.78** (still below gate)
  - **80% conversion (19/24): 40/50 = 0.80** (meets gate)

### Minimum practical calibration to hit gate

1. Prioritize citation-marker generation reliability (primary lever).
   - Target: convert at least **19 of 24** `FAIL_NO_CITATIONS` cases to citation-bearing answers that pass verification.
2. Secondary cleanup: fix the 4 unresolvable citation mapping issues.
   - This gives extra margin above 0.80 once citation presence is fixed.
3. Do not treat overlap-threshold tuning as primary.
   - `FAIL_UNSUPPORTED_CLAIMS` is only **1 case**, so impact is minimal.

Decision: the smallest credible path is a citation-presence calibration aimed at **>=80% recovery of current no-citation fails**, with first focus on `table` and `multi_page` prompts.
