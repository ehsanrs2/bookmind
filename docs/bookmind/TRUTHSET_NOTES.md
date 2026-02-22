# Truth Set Notes

Created `tools/bookmind_bench/samples/queries_scanned_tech_truth25.json` from `tools/bookmind_bench/samples/queries_scanned_tech.json` and added page-level truth labels for all currently present queries.

## How pages were determined

- Primary evidence source: existing eval artifacts in `/tmp/bookmind_eval_after_fix/items/*.json` and `/tmp/bookmind_layout_run/eval_ollama_fixed/items/*.json` (citation pages and stable IDs per query).
- Cross-check source: live Qdrant payload inspection from collection `bookmind_bench` (scrolled to `/tmp/bookmind_points.json`) to verify each cited stable ID's exact `page`, `content_type`, and snippet text.
- Additional check: `qdrant-search` runs for each query with top-k results to confirm page concentration and whether answers were discrete pages (`truth_pages`) or consecutive spans (`truth_page_ranges`).

## Scope note

The current source file `tools/bookmind_bench/samples/queries_scanned_tech.json` contains 8 queries (not 25), so this truth file annotates the first 8 available queries without inventing additional query entries.

## Truth50 coverage

Created `tools/bookmind_bench/samples/queries_scanned_tech_truth50.json` by extending the prior truth set to 50 total queries with required truth labels and coverage categories.

- Evidence basis for added queries:
  - `/tmp/bookmind_points.json` payload export (78 rows from `bookmind_bench`) for direct page/text grounding.
  - Existing eval artifacts in `/tmp/bookmind_eval_after_fix/items/*.json` and `/tmp/bookmind_layout_run/eval_ollama_fixed/items/*.json` for citation/page corroboration.
  - Spot verification with `qdrant-search` on representative added queries.
- Category counts (Truth50):
  - `text`: 19
  - `figure_caption`: 11
  - `table`: 11
  - `multi_page`: 7
  - `not_in_book`: 2
