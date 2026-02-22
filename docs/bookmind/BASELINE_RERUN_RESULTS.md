# Baseline Rerun Results (`20260220_sample1-5`)

## Run notes
- Requested E2E wrapper command was attempted, but failed at extraction because `marker_single` is not available in this environment.
- Rerun was completed using the same baseline inputs by:
  - re-ingesting baseline run bundle into `bookmind_bench`,
  - running `retrieval-eval` and `eval` with `sample/sample1-5.pdf` baseline configuration,
  - running explicit `reliability-pack` and `compare-baseline`.

Artifacts:
- Rerun eval report: `/tmp/bookmind_e2e_rerun_20260220_sample1-5/eval/report.json`
- Rerun retrieval report: `/tmp/bookmind_e2e_rerun_20260220_sample1-5/retrieval_eval/retrieval_report.json`
- Rerun reliability report: `/tmp/bookmind_reliability_pack_rerun/reliability_report.json`
- Compare report: `/tmp/bookmind_compare_rerun/compare_report.json`

## Old vs new (baseline vs rerun)

| metric | baseline | rerun |
|---|---:|---:|
| `verification_pass_rate` | `0.0` | `0.42` |
| `citations_resolvable_rate` | `1.0` | `0.92` |
| `meets_gate` | `false` | `false` |
| `recommendation` | `INVESTIGATE_VERIFIER` | `INVESTIGATE_VERIFIER` |
| `reliability_score` | `61.356` | `74.456` |

## Compare-baseline result
- `verdict`: `FAIL_GATE`
- Key deltas:
  - `reliability_score`: `+13.1`
  - `verification_pass_rate`: `+0.42`
  - `citations_resolvable_rate`: `-0.08`
  - `hit_rate@k`: `0.0`
  - `mrr`: `0.0`
- `regression_failures`: none
