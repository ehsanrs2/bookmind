# Gate Rerun Results (Force Citations, Truth50)

Run date: 2026-02-22

Artifacts:
- Retrieval eval: `/tmp/bookmind_gate_rerun/retrieval_eval/retrieval_report.json`
- Eval: `/tmp/bookmind_gate_rerun/eval/report.json`
- Reliability: `/tmp/bookmind_gate_rerun/reliability/reliability_report.json`
- Compare: `/tmp/bookmind_gate_rerun/compare/compare_report.json`

## Key results

- `verification_pass_rate`: **0.68**
- `FAIL_NO_CITATIONS`: **9**
- `FAIL_UNRESOLVABLE_CITATIONS`: **7**
- `citations_resolvable_rate`: **0.86**
- `meets_gate`: **false**
- `recommendation`: **INVESTIGATE_CITATIONS**
- `compare-baseline verdict`: **FAIL_GATE**

## Notes

- Force-citation eval improved citation presence (no-citation failures reduced), but gate still fails due citation resolvability and overall verifier threshold.
- Retrieval remained strong (`hit_rate@k=0.96`, `mrr=0.87804`) from `/tmp/bookmind_gate_rerun/retrieval_eval/retrieval_report.json`.
