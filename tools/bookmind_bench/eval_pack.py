"""Offline evaluation harness for bench RAG preview regressions."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

from engines.vlm_providers import build_vlm_provider
from qdrant_ingest import search_query
from rag_preview import build_context, build_rag_messages, generate_answer

FIGURE_BOOST_TOKENS = (
    "figure",
    "diagram",
    "schematic",
    "block diagram",
    "circuit",
    "wiring",
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def load_queries(path: str | Path) -> List[Dict[str, Any]]:
    query_path = Path(path)
    suffix = query_path.suffix.lower()
    raw = query_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YAML query files require PyYAML. Use JSON input or install PyYAML."
            ) from exc
        parsed = yaml.safe_load(raw)
    else:
        parsed = json.loads(raw)

    if not isinstance(parsed, list):
        raise ValueError("Query file must contain a list of query objects.")

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Query item #{idx} must be an object.")
        query_id = str(item.get("id") or f"q{idx}").strip()
        query_text = str(item.get("query") or "").strip()
        if not query_id:
            raise ValueError(f"Query item #{idx} has empty id.")
        if not query_text:
            raise ValueError(f"Query item #{idx} has empty query.")
        expected_pages = item.get("expected_pages")
        if expected_pages is not None and not isinstance(expected_pages, list):
            raise ValueError(f"Query item #{idx} expected_pages must be a list when present.")
        out.append(
            {
                "id": query_id,
                "query": query_text,
                "notes": str(item.get("notes") or "").strip() or None,
                "expected_pages": expected_pages,
            }
        )
    return out


def reorder_results_for_query(
    query: str,
    results: Sequence[Dict[str, Any]],
    heuristic_boost_figures: bool = True,
) -> List[Dict[str, Any]]:
    ordered = list(results)
    if not heuristic_boost_figures:
        return ordered

    lowered = str(query or "").lower()
    if not any(token in lowered for token in FIGURE_BOOST_TOKENS):
        return ordered

    figure_rows = [row for row in ordered if str(row.get("content_type") or "") == "figure_caption"]
    non_figure_rows = [row for row in ordered if str(row.get("content_type") or "") != "figure_caption"]
    return figure_rows + non_figure_rows


def _split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", str(text or "").strip())
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def _citation_rate(answer_text: str) -> float:
    sentences = _split_sentences(answer_text)
    if not sentences:
        return 0.0
    marker = re.compile(r"\[[^\[\]:]+:[^\[\]]+\]")
    cited = sum(1 for sentence in sentences if marker.search(sentence))
    return cited / len(sentences)


def run_eval(
    queries: Sequence[Dict[str, Any]],
    qdrant_url: str,
    collection: str,
    backend_cfg: Dict[str, Any],
    retrieval_cfg: Dict[str, Any],
    gen_cfg: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    items_dir = out_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    content_types = retrieval_cfg.get("content_types")
    cleaned_types = [str(item).strip() for item in (content_types or []) if str(item).strip()]
    top_k = int(retrieval_cfg.get("top_k", 8))
    max_context_chars = int(retrieval_cfg.get("max_context_chars", 6000))
    heuristic_boost_figures = bool(retrieval_cfg.get("heuristic_boost_figures", True))

    provider = build_vlm_provider(
        backend=str(backend_cfg.get("backend") or "ollama"),
        endpoint=str(backend_cfg.get("endpoint") or "http://127.0.0.1:8000/v1"),
        model=str(backend_cfg.get("model") or "qwen3-vl"),
        ollama_url=str(backend_cfg.get("ollama_url") or "http://127.0.0.1:11434"),
        ollama_model=str(backend_cfg.get("ollama_model") or "qwen3-vl:latest"),
    )

    rows: List[Dict[str, Any]] = []
    total_values: List[float] = []
    answer_len_values: List[int] = []
    citation_count_values: List[int] = []
    any_citation_count = 0
    hit_pages_values: List[int] = []

    for index, query_item in enumerate(queries, start=1):
        query_id = str(query_item.get("id") or f"q{index}")
        query_text = str(query_item.get("query") or "")
        expected_pages = query_item.get("expected_pages")

        t0 = time.perf_counter()
        retrieved = search_query(
            query=query_text,
            qdrant_url=qdrant_url,
            collection=collection,
            embed_model=str(retrieval_cfg.get("embed_model") or "all-MiniLM-L6-v2"),
            top_k=top_k,
            timeout_s=int(retrieval_cfg.get("timeout_s", 30)),
            content_types=cleaned_types if cleaned_types else None,
        )
        t1 = time.perf_counter()

        retrieved = reorder_results_for_query(
            query=query_text,
            results=retrieved,
            heuristic_boost_figures=heuristic_boost_figures,
        )

        context_payload = build_context(retrieved, max_chars=max_context_chars)
        context_text = str(context_payload.get("context_text") or "")
        citations = context_payload.get("citations")
        if not isinstance(citations, list):
            citations = []
        messages = build_rag_messages(query_text, context_text)

        t2 = time.perf_counter()
        answer_text, usage = generate_answer(
            provider=provider,
            messages=messages,
            max_tokens=int(gen_cfg.get("max_tokens", 512)),
            temperature=float(gen_cfg.get("temperature", 0.2)),
        )
        t3 = time.perf_counter()

        retrieval_ms = (t1 - t0) * 1000.0
        generation_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

        citation_rows: List[Dict[str, Any]] = []
        citation_pages: List[int] = []
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            page_value = citation.get("page")
            stable_id = str(citation.get("stable_id") or "")
            entry = {
                "page": page_value if isinstance(page_value, int) else None,
                "stable_id": stable_id,
                "content_type": citation.get("content_type"),
            }
            citation_rows.append(entry)
            if isinstance(page_value, int):
                citation_pages.append(page_value)

        retrieved_ids = [str(row.get("stable_id") or row.get("id") or "") for row in retrieved]
        answer_len_chars = len(str(answer_text or ""))
        citation_count = len(citation_rows)
        citation_rate = _citation_rate(str(answer_text or ""))

        if citation_count > 0:
            any_citation_count += 1
        if isinstance(expected_pages, list) and expected_pages:
            expected_page_set = {int(page) for page in expected_pages if isinstance(page, int)}
            hit_pages_values.append(1 if expected_page_set.intersection(citation_pages) else 0)

        row = {
            "id": query_id,
            "query": query_text,
            "notes": query_item.get("notes"),
            "expected_pages": expected_pages,
            "timings": {
                "retrieval_ms": round(retrieval_ms, 3),
                "generation_ms": round(generation_ms, 3),
                "total_ms": round(total_ms, 3),
            },
            "answer": answer_text,
            "answer_len_chars": answer_len_chars,
            "citation_rate": round(citation_rate, 4),
            "citation_count": citation_count,
            "citations": citation_rows,
            "retrieved_ids": retrieved_ids,
            "usage": usage,
        }
        rows.append(row)
        _write_json(items_dir / f"{query_id}.json", row)

        total_values.append(total_ms)
        answer_len_values.append(answer_len_chars)
        citation_count_values.append(citation_count)

    count_queries = len(rows)
    summary: Dict[str, Any] = {
        "count_queries": count_queries,
        "avg_total_ms": round(sum(total_values) / count_queries, 3) if count_queries else 0.0,
        "p50_total_ms": round(float(median(total_values)), 3) if total_values else 0.0,
        "avg_answer_len_chars": round(sum(answer_len_values) / count_queries, 3)
        if count_queries
        else 0.0,
        "avg_num_citations": round(sum(citation_count_values) / count_queries, 3)
        if count_queries
        else 0.0,
        "percent_queries_with_any_citation": round((any_citation_count / count_queries) * 100.0, 3)
        if count_queries
        else 0.0,
    }
    if hit_pages_values:
        summary["hit@k_pages"] = round((sum(hit_pages_values) / len(hit_pages_values)) * 100.0, 3)
        summary["expected_pages_evaluable_queries"] = len(hit_pages_values)

    return {
        "config": {
            "qdrant_url": qdrant_url,
            "collection": collection,
            "backend": backend_cfg,
            "retrieval": {
                "top_k": top_k,
                "content_types": cleaned_types,
                "max_context_chars": max_context_chars,
                "heuristic_boost_figures": heuristic_boost_figures,
                "embed_model": str(retrieval_cfg.get("embed_model") or "all-MiniLM-L6-v2"),
            },
            "generation": {
                "max_tokens": int(gen_cfg.get("max_tokens", 512)),
                "temperature": float(gen_cfg.get("temperature", 0.2)),
            },
        },
        "summary": summary,
        "queries": rows,
    }


def _report_lines(results: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    config = results.get("config") if isinstance(results.get("config"), dict) else {}
    summary = results.get("summary") if isinstance(results.get("summary"), dict) else {}
    queries = results.get("queries") if isinstance(results.get("queries"), list) else []

    lines.append("# Bookmind Bench Evaluation Report")
    lines.append("")
    lines.append("## Run configuration")
    lines.append(f"- qdrant_url: {config.get('qdrant_url')}")
    lines.append(f"- collection: {config.get('collection')}")
    lines.append(f"- backend: {json.dumps(config.get('backend'), ensure_ascii=True)}")
    lines.append(f"- retrieval: {json.dumps(config.get('retrieval'), ensure_ascii=True)}")
    lines.append(f"- generation: {json.dumps(config.get('generation'), ensure_ascii=True)}")
    lines.append("")
    lines.append("## Summary")
    for key in (
        "count_queries",
        "avg_total_ms",
        "p50_total_ms",
        "avg_answer_len_chars",
        "avg_num_citations",
        "percent_queries_with_any_citation",
        "hit@k_pages",
        "expected_pages_evaluable_queries",
    ):
        if key in summary:
            lines.append(f"- {key}: {summary.get(key)}")

    lines.append("")
    lines.append("## Per-query")
    for row in queries:
        if not isinstance(row, dict):
            continue
        timings = row.get("timings") if isinstance(row.get("timings"), dict) else {}
        lines.append(
            f"- {row.get('id')}: total_ms={timings.get('total_ms')} "
            f"citations={row.get('citation_count')} citation_rate={row.get('citation_rate')} "
            f"answer_len={row.get('answer_len_chars')}"
        )
        lines.append(f"  query: {row.get('query')}")
        lines.append(f"  retrieved_ids: {row.get('retrieved_ids')}")
        lines.append(f"  citations: {row.get('citations')}")

    sorted_by_citations = sorted(
        [row for row in queries if isinstance(row, dict)],
        key=lambda row: int(row.get("citation_count") or 0),
    )
    sorted_by_latency = sorted(
        [row for row in queries if isinstance(row, dict)],
        key=lambda row: float(
            (row.get("timings") or {}).get("total_ms") if isinstance(row.get("timings"), dict) else 0.0
        ),
        reverse=True,
    )

    lines.append("")
    lines.append("## Worst 3 by citation_count")
    for row in sorted_by_citations[:3]:
        timings = row.get("timings") if isinstance(row.get("timings"), dict) else {}
        lines.append(
            f"- {row.get('id')}: citation_count={row.get('citation_count')} "
            f"total_ms={timings.get('total_ms')}"
        )

    lines.append("")
    lines.append("## Slowest 3 by total_ms")
    for row in sorted_by_latency[:3]:
        timings = row.get("timings") if isinstance(row.get("timings"), dict) else {}
        lines.append(
            f"- {row.get('id')}: total_ms={timings.get('total_ms')} "
            f"citation_count={row.get('citation_count')}"
        )

    lines.append("")
    return lines


def write_reports(results_dict: Dict[str, Any], output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"
    _write_json(report_json, results_dict)
    report_md.write_text("\n".join(_report_lines(results_dict)), encoding="utf-8")
