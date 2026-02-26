# BookMind Ask API (Local)

Minimal FastAPI wrapper for the existing `run.py ask` path.

## Install

Install bench deps (now includes FastAPI + Uvicorn):

```bash
pip install -r tools/bookmind_bench/requirements/base.txt
```

## Run

```bash
python tools/bookmind_bench/run.py serve \
  --host 127.0.0.1 \
  --port 8001 \
  --qdrant_url http://127.0.0.1:6333 \
  --collection bookmind_bench \
  --backend ollama \
  --ollama_url http://127.0.0.1:11434 \
  --ollama_model qwen3-vl:latest \
  --ollama_api generate
```

## Endpoints

### `GET /health`

Returns:

```json
{"status":"ok"}
```

May also include:
- `version.git_commit` when git metadata is available.
- `qdrant` connectivity info when `qdrant_url` is configured.

### `POST /ask`

Request JSON:

```json
{
  "query": "What does the TTAS channel do?",
  "qdrant_url": "http://127.0.0.1:6333",
  "collection": "bookmind_bench",
  "top_k": 20,
  "content_types": ["text", "figure_caption"],
  "backend": "ollama",
  "ollama_url": "http://127.0.0.1:11434",
  "ollama_model": "qwen3-vl:latest",
  "ollama_api": "generate",
  "force_citations": false,
  "citation_min_count": 1,
  "citation_repair_retry": "auto",
  "enforce_verified": false
}
```

Behavior:
- Response body is exactly the same JSON schema as `run.py ask`.
- If `enforce_verified=true` and `status != "PASS"`, HTTP status is `422` and body is unchanged payload.
- Otherwise HTTP status is `200`.

## curl examples

Health:

```bash
curl -s http://127.0.0.1:8001/health
```

Ask:

```bash
curl -s -X POST http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the TTAS channel do?",
    "backend": "ollama",
    "ollama_api": "generate"
  }'
```

Ask with strict verification (returns `422` on non-PASS):

```bash
curl -s -o /tmp/bookmind_ask.json -w "%{http_code}\n" \
  -X POST http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the TTAS channel do?",
    "enforce_verified": true
  }'
```

## OpenWebUI usage

Point your integration/client to:
- URL: `http://127.0.0.1:8001/ask`
- Method: `POST`
- Content type: `application/json`

Use `query` from user message and map the returned JSON fields directly in UI logic.
