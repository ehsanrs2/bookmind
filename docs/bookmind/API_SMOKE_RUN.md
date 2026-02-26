# BookMind API Smoke Run

## Environment
- Repo: `/media/ehsan/SSD1TB/Bookmind`
- API server bind: `127.0.0.1:8001`
- Qdrant: `http://127.0.0.1:6333`
- Ollama: `http://127.0.0.1:11434`

## 1) Start Server

Command used:

```bash
./tools/bookmind_bench/scripts/bench_python.sh tools/bookmind_bench/run.py serve --host 127.0.0.1 --port 8001
```

## 2) Health Check

Command used:

```bash
curl -s http://127.0.0.1:8001/health | jq .
```

Output:

```json
{
  "status": "ok",
  "version": {
    "git_commit": "5168691cf"
  },
  "qdrant": {
    "ok": true,
    "url": "http://127.0.0.1:6333"
  }
}
```

## 3) Ask Check (PASS Query)

PASS query used (from baseline `A_compass_output`):
- `What output does the standby magnetic compass provide?`

Command used:

```bash
curl -s -X POST http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query":"What output does the standby magnetic compass provide?",
    "qdrant_url":"http://127.0.0.1:6333",
    "collection":"bookmind_bench",
    "backend":"ollama",
    "ollama_url":"http://127.0.0.1:11434",
    "ollama_model":"qwen3-vl:latest",
    "ollama_api":"generate",
    "force_citations":true,
    "citation_min_count":1,
    "citation_repair_retry":"auto",
    "enforce_verified":false
  }' | jq .
```

Ask output summary:
- `status`: `PASS`
- `verification_status`: `PASS`
- `citations_total`: `2`
- `citations_resolved`: `2`
- `citations_resolvable`: `true`
