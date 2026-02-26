from __future__ import annotations

import json

from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

import api_server


def _endpoint(app, path: str, method: str):
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path and method.upper() in (route.methods or set()):
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


def test_ask_returns_200_and_payload_unchanged(monkeypatch) -> None:
    expected = {
        "query": "what is x",
        "final_answer": "answer",
        "status": "PASS",
        "verification_status": "PASS",
    }
    monkeypatch.setattr(api_server.run, "_run_ask_query", lambda args, *, query: expected)

    app = api_server.create_app(qdrant_url="")
    ask_endpoint = _endpoint(app, "/ask", "POST")
    response = ask_endpoint(api_server.AskRequest(query="what is x"))

    assert response == expected


def test_ask_enforce_verified_returns_422_and_payload_unchanged(monkeypatch) -> None:
    expected = {
        "query": "what is x",
        "final_answer": "answer",
        "status": "FAIL",
        "verification_status": "FAIL_UNRESOLVABLE_CITATIONS",
    }
    monkeypatch.setattr(api_server.run, "_run_ask_query", lambda args, *, query: expected)

    app = api_server.create_app(qdrant_url="")
    ask_endpoint = _endpoint(app, "/ask", "POST")
    response = ask_endpoint(api_server.AskRequest(query="what is x", enforce_verified=True))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 422
    assert json.loads(response.body.decode("utf-8")) == expected


def test_health_returns_ok() -> None:
    app = api_server.create_app(qdrant_url="")
    health_endpoint = _endpoint(app, "/health", "GET")
    response = health_endpoint()

    assert response["status"] == "ok"
