# Product Baseline (ask)

This folder snapshots a product smoke baseline for the `ask` interface.

## Source Context

- Bundle path: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/bundle`
- Bundle records file: `/tmp/bookmind_e2e_baseline_20260220_sample1-5/bundle/records.jsonl`
- Bundle records SHA-256: `120318ae44a5f7958d7dc953341d7ad455846c989596b68973cf09ab08f1cc2c`
- Qdrant URL: `http://127.0.0.1:6333`
- Qdrant collection: `bookmind_bench`
- Backend: `ollama`
- Model: `qwen3-vl:latest`
- Ollama API mode: `generate`

## Expected Statuses

- A (`ask_A.json`): `PASS`
- B (`ask_B.json`): `PASS`
- C (`ask_C.json`): `NOT_FOUND`

## Files

- `PRODUCT_SMOKE_RUN.md`
- `E2E_SUMMARY.md`
- `ask_A.json`
- `ask_B.json`
- `ask_C.json`
