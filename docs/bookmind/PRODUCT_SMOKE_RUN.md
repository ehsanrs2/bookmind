# Product Smoke Run

- Run date: 2026-02-25
- Repo: `/media/ehsan/SSD1TB/Bookmind`
- E2E command mode: `--from_bundle`
- Bundle used: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/bundle`
- E2E out: `/tmp/bookmind_product_smoke`
- Qdrant: `http://127.0.0.1:6333`
- Collection: `bookmind_bench`

## Query Results

| Case | Truth50 ID | Category | Query | Status | citations_total | citations_resolved | citations_resolvable | verification_status | Output |
|---|---|---|---|---|---:|---:|---|---|---|
| A | `q12_compass_output` | `text` | What output does the standby magnetic compass provide? | `PASS` | 2 | 2 | `true` | `PASS` | `/tmp/bookmind_product_smoke/ask_A.json` |
| B | `q47_aoa_airdata_crosspage` | `multi_page` | Summarize how angle-of-attack and air data topics connect across sections. | `PASS` | 1 | 1 | `true` | `PASS` | `/tmp/bookmind_product_smoke/ask_B.json` |
| C | `q49_notin_hydraulic_pressure` | `not_in_book` | What hydraulic pressure setpoints are specified for the landing gear system? | `NOT_FOUND` | 0 | 0 | `true` | `FAIL_NO_CITATIONS` | `/tmp/bookmind_product_smoke/ask_C.json` |

## Enforce Verified Check (Case C)

- Command: same as Case C plus `--enforce_verified true`
- Output JSON: `/tmp/bookmind_product_smoke/ask_C_verified.json`
- Exit code: `2`
- Result summary: `status=NOT_FOUND`, `verification_status=FAIL_NO_CITATIONS`

## Unexpected Behavior

- `tools/bookmind_bench/scripts/e2e_bookmind.sh --from_bundle ...` completed ingest successfully (`qdrant-ingest: RUN`, exit code `0`) but printed multiple shell errors near summary generation, e.g. `from_bundle\: command not found` and `No such file or directory` lines for paths that do exist.
- `/tmp/bookmind_product_smoke/E2E_SUMMARY.md` was generated with blank `\` values for many fields (`mode`, `from_bundle`, `out`, etc.), suggesting an escaping/templating bug in the summary section of the script.

## Deliverables

- `/tmp/bookmind_product_smoke/ask_A.json`
- `/tmp/bookmind_product_smoke/ask_B.json`
- `/tmp/bookmind_product_smoke/ask_C.json`
- `/tmp/bookmind_product_smoke/ask_C_verified.json`
- `docs/bookmind/PRODUCT_SMOKE_RUN.md`
