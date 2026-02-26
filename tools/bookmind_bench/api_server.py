"""Minimal FastAPI server for Bookmind ask endpoint."""

from __future__ import annotations

import argparse
import subprocess
from typing import Literal
from urllib import error, request

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import run
from verifier import PASS


class AskRequest(BaseModel):
    query: str
    qdrant_url: str = "http://127.0.0.1:6333"
    collection: str = "bookmind_bench"
    top_k: int = 20
    content_types: list[str] = Field(default_factory=lambda: ["text", "figure_caption"])
    backend: str = "ollama"
    ollama_url: str | None = None
    ollama_model: str | None = None
    ollama_api: str = "generate"
    force_citations: bool = False
    citation_min_count: int = 1
    citation_repair_retry: Literal["auto", "true", "false"] = "auto"
    enforce_verified: bool = False


def _git_commit_short() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    commit = proc.stdout.strip()
    return commit or None


def _qdrant_health(qdrant_url: str) -> dict:
    endpoint = f"{qdrant_url.rstrip('/')}/collections"
    try:
        with request.urlopen(endpoint, timeout=2.0) as response:
            status = int(getattr(response, "status", 0) or 0)
        return {"ok": 200 <= status < 300, "url": qdrant_url}
    except error.URLError as exc:
        return {"ok": False, "url": qdrant_url, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "url": qdrant_url, "error": str(exc)}


def _pick_value(req: AskRequest, field_name: str, defaults: dict[str, object]) -> object:
    if field_name in req.model_fields_set:
        return getattr(req, field_name)
    if field_name in defaults:
        return defaults[field_name]
    return getattr(req, field_name)


def _build_ask_args(req: AskRequest, defaults: dict[str, object]) -> argparse.Namespace:
    parser = run.build_parser()
    args = parser.parse_args(["ask", "--query", req.query])

    args.query = req.query
    args.qdrant_url = str(_pick_value(req, "qdrant_url", defaults))
    args.collection = str(_pick_value(req, "collection", defaults))
    args.top_k = int(_pick_value(req, "top_k", defaults))
    args.content_types = ",".join(str(item) for item in _pick_value(req, "content_types", defaults))
    args.backend = str(_pick_value(req, "backend", defaults))
    args.ollama_api = str(_pick_value(req, "ollama_api", defaults))
    args.force_citations = bool(_pick_value(req, "force_citations", defaults))
    args.citation_min_count = int(_pick_value(req, "citation_min_count", defaults))
    args.citation_repair_retry = str(_pick_value(req, "citation_repair_retry", defaults))
    args.enforce_verified = bool(_pick_value(req, "enforce_verified", defaults))

    ollama_url = _pick_value(req, "ollama_url", defaults)
    if ollama_url is not None:
        args.ollama_url = str(ollama_url)
    ollama_model = _pick_value(req, "ollama_model", defaults)
    if ollama_model is not None:
        args.ollama_model = str(ollama_model)

    return args


def create_app(
    *,
    qdrant_url: str | None = "http://127.0.0.1:6333",
    collection: str = "bookmind_bench",
    backend: str = "ollama",
    ollama_url: str | None = None,
    ollama_model: str | None = None,
    ollama_api: str = "generate",
) -> FastAPI:
    app = FastAPI(title="Bookmind Bench API")
    defaults: dict[str, object] = {
        "qdrant_url": qdrant_url,
        "collection": collection,
        "backend": backend,
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
        "ollama_api": ollama_api,
    }

    @app.get("/health")
    def health() -> dict:
        payload: dict[str, object] = {"status": "ok"}
        commit = _git_commit_short()
        if commit:
            payload["version"] = {"git_commit": commit}
        configured_qdrant = defaults.get("qdrant_url")
        if isinstance(configured_qdrant, str) and configured_qdrant.strip():
            payload["qdrant"] = _qdrant_health(configured_qdrant)
        return payload

    @app.post("/ask")
    def ask(req: AskRequest):
        ask_args = _build_ask_args(req, defaults)
        result = run._run_ask_query(ask_args, query=req.query)
        if bool(ask_args.enforce_verified) and str(result.get("status") or "") != PASS:
            return JSONResponse(status_code=422, content=result)
        return result

    return app


app = create_app()
