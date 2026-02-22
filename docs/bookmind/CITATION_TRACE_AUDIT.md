# Citation Trace Audit (`baseline_20260220_sample1-5`)

## Scope
- Eval dir: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/eval`
- Report: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/eval/report.json`
- Queries audited:
  - text: `q24_pressure_difference_p`
  - table: `q44_tas_repeater_set`

## Query 1: `q24_pressure_difference_p` (text)

### Artifacts found
- Answer text: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/eval/per_query/q24_answer.txt`
- Raw saved model response: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/eval/items/q24_pressure_difference_p.json` (`answer_text`/`answer`)
- Per-query citation artifact: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/eval/per_query/q24_citations.json`
- Retrieved context listing saved by eval pack: same `q24_citations.json` under `citations` + `retrieved_ids`

### Exact `answer_text` as stored (first 20 lines)
```text
FINAL ANSWER  
Pressure p for the pitot-static aneroid is defined as the difference between the pitot pressure (inside the aneroid) and the static pressure (inside the case), acting on the aneroid to cause deformation. [page:32522d73d6d3a42c] [page:94080d280f711874]  

- p represents the differential pressure (pitot minus static) that distorts the aneroid.  
- This differential pressure is critical for the pitot-static aneroid's function in airspeed and Mach number measurement.
```

### Trace values
- `parse_citations(answer_text)` count: `0`
- Retrieved context citations list count (saved): `8`
- Per-query artifact fields:
  - `citations_total=0`
  - `citations_resolved=0`
  - `citations_resolvable=true`
  - `verification_status=FAIL_NO_CITATIONS`
  - `verification_reason="Model answer has no citation markers."`

## Query 2: `q44_tas_repeater_set` (table)

### Artifacts found
- Answer text: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/eval/per_query/q44_answer.txt`
- Raw saved model response: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/eval/items/q44_tas_repeater_set.json` (`answer_text`/`answer`)
- Per-query citation artifact: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/eval/per_query/q44_citations.json`
- Retrieved context listing saved by eval pack: same `q44_citations.json` under `citations` + `retrieved_ids`

### Exact `answer_text` as stored (first 20 lines)
```text
FINAL ANSWER: Repeater potentiometers listed for TAS are TAS2, TAS3, TAS4, and TAS5 [page:5, stable_id:0b3dcdd038dd0db6].  
- TAS2, TAS3, TAS4, and TAS5 are explicitly identified as repeater potentiometers for TAS in the context [page:5, stable_id:0b3dcdd038dd0db6].  
- TAS computation involves potentiometers like TAS2/TAS3/TAS4/TAS5, with TAS = f(M Tt) functions utilizing specific wiper signals [page:5, stable_id:2d592b20459ddd22].
```

### Trace values
- `parse_citations(answer_text)` count: `0`
- Retrieved context citations list count (saved): `8`
- Per-query artifact fields:
  - `citations_total=0`
  - `citations_resolved=0`
  - `citations_resolvable=true`
  - `verification_status=FAIL_NO_CITATIONS`
  - `verification_reason="Model answer has no citation markers."`

## Where `avg_num_citations` comes from
- In eval pack, `citations` is taken from `build_context(retrieved, ...)` (context-side rows), not parsed model markers: `tools/bookmind_bench/eval_pack.py:264`, `tools/bookmind_bench/eval_pack.py:266`.
- `citation_rows` and `citation_count` are built from that context citation list: `tools/bookmind_bench/eval_pack.py:332`, `tools/bookmind_bench/eval_pack.py:350`.
- Summary `avg_num_citations` is computed from `citation_count_values` (context counts): `tools/bookmind_bench/eval_pack.py:463`.
- Verifier path uses parsed model citations from `answer_text`: `parse_citations(answer_text)` then `resolve_citations(...)`: `tools/bookmind_bench/eval_pack.py:352`, `tools/bookmind_bench/eval_pack.py:353`.

## Contract mismatch evidence
- Parser expects bracket body left token to be an integer page (`left` parsed by `_to_int`) after split on first `:`: `tools/bookmind_bench/citations.py:37`, `tools/bookmind_bench/citations.py:43`.
- For q24 markers like `[page:32522d73d6d3a42c]`, `left="page"` (not int) => dropped.
- For q44 markers like `[page:5, stable_id:0b3dc... ]`, split gives `left="page"`, `right="5, stable_id:..."` => dropped.
- Result: parser returns `[]` for both audited answers, so verifier receives `citations_total=0` and emits `FAIL_NO_CITATIONS`.

## Diagnosis
`report.json` is mixing two citation notions:
- `avg_num_citations` is based on **retrieved context citations**.
- `verification_status`/`citations_total` are based on **parsed model citations**.

The model outputs citation-like text, but in a format not accepted by `parse_citations`, so model citation parsing is always zero.  
Therefore: **report counts context citations, verifier checks model citations, and the model citation format does not satisfy the parser contract**.
