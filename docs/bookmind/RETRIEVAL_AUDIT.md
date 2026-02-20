# Retrieval Architecture Audit

Date: 2026-02-20  
Scope: `tools/bookmind_bench/*` bench retrieval path (extract -> normalize -> bundle -> embed -> Qdrant -> retrieve -> RAG preview).

## What exists today

### Pipeline entrypoints and handoff contracts

- CLI entrypoint: `tools/bookmind_bench/run.py` (`build_parser`, `main`).
- Implemented subcommands relevant to retrieval:
  - Extraction: `marker`, `paddleocr`, `layout`, `plan-vlm`, `vlm`, `vlm-layout`
  - Normalization/ingest prep: `merge`, `bundle`
  - Vector ingest/retrieval: `qdrant-ingest`, `qdrant-search`
  - Retrieval+generation: `rag-preview`, `eval`

### Record schemas in use

- Contract doc: `tools/bookmind_bench/schema.md`
  - Defines per-engine JSONL outputs and normalized `ingest_record`.
- Actual emitters:
  - Marker page records: `engines/marker_engine.py::run_marker_pdf`, `_normalize_marker_pages`
  - Paddle records: `engines/paddleocr_engine.py::run_paddleocr`, `_build_record`
  - Layout records: `engines/layout_engine.py::run_layoutparser`, `_build_record`
  - VLM records: `engines/vlm_caption_engine.py::run_vlm_caption_jobs`, `run_vlm_layout`
  - Ingest records: `merge.py::merge_outputs`, `_build_ingest_record`

### Where chunking, embeddings, Qdrant upsert happen

- Chunking:
  - No semantic chunking stage exists.
  - Retrieval units are region-level normalized rows from `merge.py`:
    - `content_type` in `{text, table, figure_caption}`
    - one vectorized unit per merged row.
- Embedding:
  - `qdrant_ingest.py::_load_embedding_model` + `embed_texts`
  - Applied to each `records.jsonl` row text in `ingest_bundle`.
- Qdrant upsert:
  - `qdrant_ingest.py::ingest_bundle`
  - Builds payload via `_build_payload`
  - Converts `stable_id` -> point id via `qdrant_point_id_from_stable_id`
  - Upserts batch points with `client.upsert(...)`.

### Retrieval and citation wiring

- Retrieval:
  - `qdrant_ingest.py::search_query` returns payload with `stable_id`, `page`, `bbox`, `content_type`, `text`, `meta`.
- Context/citations:
  - `rag_preview.py::build_context` builds snippet headers and citations list from retrieved rows.
  - Citation key is `(page, stable_id)`.
- Preview formatting:
  - `rag_preview.py::format_preview_output` prints citation rows with `stable_id`.

## Implemented vs missing for full retrieval architecture

### 1) `document_id` / `source_id` scheme

- Implemented:
  - `stable_id` deterministic per merged row in `merge.py::_stable_id`.
  - `bundle` manifest stores run-level provenance in `bundle.py` (`manifest["sources"]`).
- Missing / risk:
  - No explicit `document_id` in merged/bundled records.
  - No `source_id`/`chunk_id` alignment to Bookmind chunk metadata schema (`backend/open_webui/schemas/bookmind/chunk-metadata.schema.json` requires `source_id`, `chunk_id`).
  - Current retrieval identity is bench-local (`stable_id`) and not namespaced by document/source.

### 2) Page indexing and stable ordering

- Implemented:
  - Pages are 1-based across engines (`page`, `pdf_page_start`, `pdf_page_end`).
  - Per-page image listing sorted deterministically:
    - `paddleocr_engine.py::list_page_images`
    - `layout_engine.py::list_page_images`
  - Layout region ordering deterministic by geometry (`layout_engine.py::run_layoutparser` sorts by top-left).
  - Merge stable-id deterministic hash over `(page, content_type, bbox_rounded, text_norm)`.
- Missing / risk:
  - No explicit global `page_index` + `region_index` field persisted in normalized ingest rows (ordering is implicit by JSONL order + trace line ids).
  - `stable_id` can change if OCR text changes (expected), but no separate position-stable region identity.

### 3) Region-level metadata (`bbox`, `confidence`, `content_type`)

- Implemented:
  - `bbox` propagated across engines and into ingest rows.
  - `confidence` captured when available (`meta.confidence`).
  - `content_type` normalized to `text|table|figure_caption` in merge output.
  - Trace metadata includes source line ids and optional layout region info (`meta.trace.*`).
- Missing / risk:
  - `confidence` may be absent; no normalization policy/threshold gating at ingest time.
  - No explicit confidence provenance per engine (single merged confidence field).

### 4) `chunk_id` determinism

- Implemented:
  - Deterministic `stable_id` (sha1 short hash) is enforced/tested (`tests/test_merge_output_schema.py`).
  - Deterministic point-id derivation for Qdrant (`qdrant_point_id_from_stable_id`).
- Missing / risk:
  - No explicit `chunk_id` field in records that matches Bookmind schema contract.
  - No versioned ID algorithm namespace (future ID recipe changes could silently fork identity).

### 5) Citation format and mapping back to stored chunks

- Implemented:
  - Citation format in prompts/output is `[page:stable_id]`.
  - `build_context` emits citations derived from retrieved rows; `stable_id` maps directly to Qdrant payload `stable_id`.
  - Figure citation can carry `figure_ref` for crop/page image provenance.
- Missing / risk:
  - Generator output citations are not parsed/validated for non-Ollama paths; preview currently prints retrieval citations from context list, not necessarily model-selected citations.
  - Citation contract is bench-specific (`stable_id`) and not yet aligned to future `source_id/chunk_id`.

## Minimal set of next changes to declare retrieval complete

1. Add canonical identity fields at merge time:
   - `document_id` (stable per PDF/source),
   - `source_id` (stable source object id),
   - `chunk_id` (deterministic, versioned recipe).
2. Keep `stable_id` temporarily as compatibility alias, but define `chunk_id` as primary key for retrieval and citations.
3. Persist explicit ordering metadata:
   - `page_index` (1-based),
   - `region_index` (stable within page),
   - optional `reading_order`.
4. Update Qdrant payload contract in `qdrant_ingest.py::_build_payload` to include identity fields above.
5. Update `rag_preview.py` citation contract to `[page:chunk_id]` (or `[source_id:chunk_id]`) and validate citations against retrieved ids in all backends.
6. Add schema-level validation gate before bundle/ingest:
   - validate against bench ingest schema + Bookmind chunk metadata schema compatibility subset.

## Definition of Done (scanned technical PDFs)

- `merge` output rows contain: `document_id`, `source_id`, `chunk_id`, `page`, `content_type`, `text`, `bbox`, `meta.confidence`, trace metadata.
- IDs are deterministic across reruns on unchanged inputs (bit-for-bit ID stability test).
- Page ordering and region ordering are explicit and deterministic.
- Figure-caption records preserve link-back metadata to page/crop and source region.
- Qdrant payload includes identity fields and can round-trip from retrieval hit -> original merged row without ambiguity.
- Citation format in answers references stored chunk identity (not only transient rank).
- Automated tests cover:
  - ID determinism,
  - schema validation,
  - retrieval round-trip mapping,
  - citation parse/validation for `rag-preview` and `eval`.
- On a representative scanned technical PDF sample:
  - non-empty retrieval results for text/table/figure queries,
  - citations resolve to real stored chunks/pages,
  - no missing required metadata fields in ingested records.

## Should we redesign the pipeline now?

No.

Rationale: the current pipeline already has the core retrieval spine (normalized records, deterministic row identity, embedding, vector ingest, retrieval, citation-bearing context). The gap is mainly contract hardening and identity alignment (`document_id/source_id/chunk_id`), not a fundamental architecture rewrite. A targeted contract-evolution pass is lower risk and should be done before any large redesign.
