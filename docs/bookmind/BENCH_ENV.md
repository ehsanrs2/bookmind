# Bench Environment

Use a dedicated bench interpreter so retrieval/eval commands always have `qdrant_client` and embedding deps.

## Canonical Bench Venv

- Canonical venv path: `.venv_bookmind_bench_qdrant`
- Canonical interpreter: `./.venv_bookmind_bench_qdrant/bin/python`

## Create the venv

Option A (recommended, uses project requirements):

```bash
python3 -m venv .venv_bookmind_bench_qdrant
./.venv_bookmind_bench_qdrant/bin/python -m pip install --upgrade pip setuptools wheel
./.venv_bookmind_bench_qdrant/bin/python -m pip install -r tools/bookmind_bench/requirements/base.txt
./.venv_bookmind_bench_qdrant/bin/python -m pip install -r tools/bookmind_bench/requirements/qdrant.txt
```

Option B (minimal manual install):

```bash
python3 -m venv .venv_bookmind_bench_qdrant
./.venv_bookmind_bench_qdrant/bin/python -m pip install --upgrade pip setuptools wheel
./.venv_bookmind_bench_qdrant/bin/python -m pip install qdrant-client sentence-transformers
```

## Run commands with explicit bench python

Qdrant search:

```bash
./.venv_bookmind_bench_qdrant/bin/python tools/bookmind_bench/run.py qdrant-search \
  --query "test query" \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench
```

Retrieval eval:

```bash
./tools/bookmind_bench/scripts/bench_python.sh tools/bookmind_bench/run.py retrieval-eval \
  --queries tools/bookmind_bench/samples/queries_scanned_tech_truth25.json \
  --out /tmp/bookmind_retrieval_eval/mixed \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --top_k 20 \
  --content_types text,figure_caption \
  --progress true
```

## Regression Gate

Use `reliability-pack` as a single regression gate artifact:

```bash
./tools/bookmind_bench/scripts/bench_python.sh tools/bookmind_bench/run.py reliability-pack \
  --retrieval_report /tmp/bookmind_retrieval_eval/mixed/retrieval_report.json \
  --eval_report /tmp/bookmind_eval_after_fix/report.json \
  --out /tmp/bookmind_reliability_pack
```

The resulting `reliability_report.json` includes `meets_gate` and a triage `recommendation`.
