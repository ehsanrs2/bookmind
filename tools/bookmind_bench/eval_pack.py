"""Offline evaluation harness for bench RAG preview regressions."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

from citations import parse_citations, resolve_citations
from engines.vlm_providers import build_vlm_provider
from qdrant_ingest import search_query
from rag_preview import (
    build_citation_repair_messages,
    build_context,
    build_force_citation_allowlist,
    build_rag_messages,
    generate_answer,
    write_ollama_attempt_artifacts,
)
from verifier import PASS, verify_answer

FIGURE_BOOST_TOKENS = (
    "figure",
    "diagram",
    "schematic",
    "block diagram",
    "circuit",
    "wiring",
)


def _is_not_found_answer(answer_text: str) -> bool:
    return str(answer_text or "").strip() == "NOT_FOUND"


@dataclass
class _ProgressReporter:
    enabled: bool
    total: int
    tqdm_bar: Any = None
    index: int = 0

    def phase(self, query_id: str, phase: str) -> None:
        if not self.enabled:
            return
        qid = str(query_id or "")
        phase_name = str(phase or "").strip()
        if self.tqdm_bar is not None:
            self.tqdm_bar.set_description_str(f"{self.index + 1}/{self.total} {qid}")
            self.tqdm_bar.set_postfix_str(phase_name)
            return
        print(f"[{self.index + 1}/{self.total}] {qid}: {phase_name}...")

    def next_query(self) -> None:
        if not self.enabled:
            return
        self.index += 1
        if self.tqdm_bar is not None:
            self.tqdm_bar.update(1)

    def close(self) -> None:
        if self.tqdm_bar is not None:
            self.tqdm_bar.close()


def _build_progress_reporter(total: int, enabled: bool) -> _ProgressReporter:
    if not enabled:
        return _ProgressReporter(enabled=False, total=total)
    try:
        from tqdm import tqdm  # type: ignore

        return _ProgressReporter(
            enabled=True,
            total=total,
            tqdm_bar=tqdm(total=max(0, int(total)), unit="query", dynamic_ncols=True),
        )
    except Exception:
        return _ProgressReporter(enabled=True, total=total)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def _write_provider_debug(
    *,
    per_query_dir: Path,
    query_prefix: str,
    query_id: str,
    issue: str,
    usage: Optional[Dict[str, Any]],
    error: Optional[Exception] = None,
) -> None:
    payload: Dict[str, Any] = {
        "id": query_id,
        "issue": issue,
        "usage": usage if isinstance(usage, dict) else None,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
    _write_json(per_query_dir / f"{query_prefix}_raw_provider.json", payload)


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
        truth_pages = item.get("truth_pages")
        if truth_pages is not None and not isinstance(truth_pages, list):
            raise ValueError(f"Query item #{idx} truth_pages must be a list when present.")
        truth_page_ranges = item.get("truth_page_ranges")
        if truth_page_ranges is not None and not isinstance(truth_page_ranges, list):
            raise ValueError(f"Query item #{idx} truth_page_ranges must be a list when present.")
        out.append(
            {
                "id": query_id,
                "query": query_text,
                "notes": str(item.get("notes") or "").strip() or None,
                "expected_pages": expected_pages,
                "truth_pages": truth_pages,
                "truth_page_ranges": truth_page_ranges,
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
    progress: bool = False,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    items_dir = out_dir / "items"
    per_query_dir = out_dir / "per_query"
    items_dir.mkdir(parents=True, exist_ok=True)
    per_query_dir.mkdir(parents=True, exist_ok=True)

    content_types = retrieval_cfg.get("content_types")
    cleaned_types = [str(item).strip() for item in (content_types or []) if str(item).strip()]
    top_k = int(retrieval_cfg.get("top_k", 8))
    max_context_chars = int(retrieval_cfg.get("max_context_chars", 6000))
    heuristic_boost_figures = bool(retrieval_cfg.get("heuristic_boost_figures", True))
    force_citations = bool(gen_cfg.get("force_citations", False))
    citation_min_count = max(1, int(gen_cfg.get("citation_min_count", 1)))
    citation_repair_retry = bool(gen_cfg.get("citation_repair_retry", force_citations))

    provider = build_vlm_provider(
        backend=str(backend_cfg.get("backend") or "ollama"),
        endpoint=str(backend_cfg.get("endpoint") or "http://127.0.0.1:8000/v1"),
        model=str(backend_cfg.get("model") or "qwen3-vl"),
        ollama_url=str(backend_cfg.get("ollama_url") or "http://127.0.0.1:11434"),
        ollama_model=str(backend_cfg.get("ollama_model") or "qwen3-vl:latest"),
        ollama_format=str(backend_cfg.get("ollama_format") or "text"),
        ollama_num_ctx=backend_cfg.get("ollama_num_ctx"),
        ollama_api=str(backend_cfg.get("ollama_api") or "chat"),
    )

    rows: List[Dict[str, Any]] = []
    total_values: List[float] = []
    answer_len_values: List[int] = []
    context_citation_count_values: List[int] = []
    model_citation_count_values: List[int] = []
    any_citation_count = 0
    verification_passes = 0
    hit_pages_values: List[int] = []
    failed_queries = 0
    backend_name = str(backend_cfg.get("backend") or "").strip().lower()
    progress_reporter = _build_progress_reporter(len(queries), enabled=bool(progress))

    try:
        for index, query_item in enumerate(queries, start=1):
            query_id = str(query_item.get("id") or f"q{index}")
            query_text = str(query_item.get("query") or "")
            expected_pages = query_item.get("expected_pages")
            per_query_prefix = f"q{index:02d}"

            progress_reporter.phase(query_id, "retrieve")
            t0 = time.perf_counter()
            retrieved = search_query(
                query=query_text,
                qdrant_url=qdrant_url,
                collection=collection,
                embed_model=str(retrieval_cfg.get("embed_model") or "all-MiniLM-L6-v2"),
                top_k=top_k,
                timeout_s=int(retrieval_cfg.get("timeout_s", 30)),
                content_types=cleaned_types if cleaned_types else None,
                embed_cache_dir=retrieval_cfg.get("embed_cache_dir"),
                embed_local_only=bool(retrieval_cfg.get("embed_local_only", False)),
                hf_timeout_s=int(retrieval_cfg.get("hf_timeout_s", 30)),
                hf_retries=int(retrieval_cfg.get("hf_retries", 3)),
            )
            t1 = time.perf_counter()

            retrieved = reorder_results_for_query(
                query=query_text,
                results=retrieved,
                heuristic_boost_figures=heuristic_boost_figures,
            )

            progress_reporter.phase(query_id, "build_context")
            context_payload = build_context(retrieved, max_chars=max_context_chars)
            context_text = str(context_payload.get("context_text") or "")
            citations = context_payload.get("citations")
            if not isinstance(citations, list):
                citations = []
            messages = build_rag_messages(
                query_text,
                context_text,
                require_json_response=False,
                force_citations=force_citations,
                citation_min_count=citation_min_count,
                allowed_citation_ids=(
                    build_force_citation_allowlist(citations) if force_citations else None
                ),
            )

            progress_reporter.phase(query_id, "generate")
            t2 = time.perf_counter()
            usage: Optional[Dict[str, Any]]
            generation_error: Optional[Exception] = None
            ollama_attempts: List[Dict[str, Any]] = []
            hook = ollama_attempts.append if backend_name == "ollama" else None
            try:
                raw_answer_text, usage = generate_answer(
                    provider=provider,
                    messages=messages,
                    max_tokens=int(gen_cfg.get("max_tokens", 512)),
                    temperature=float(gen_cfg.get("temperature", 0.2)),
                    ollama_num_predict=int(gen_cfg.get("ollama_num_predict", 1536)),
                    ollama_retry_num_predict=int(gen_cfg.get("ollama_retry_num_predict", 2048)),
                    max_context_chars=int(gen_cfg.get("max_context_chars", max_context_chars)),
                    ollama_num_predict_auto=bool(gen_cfg.get("ollama_num_predict_auto", True)),
                    ollama_think=gen_cfg.get("ollama_think", False),
                    fallback_citations=citations,
                    ollama_debug_hook=hook,
                )
            except Exception as exc:
                raw_answer_text = ""
                usage = None
                generation_error = exc

            answer_text = str(raw_answer_text or "").strip()
            citation_repair_applied = False
            citation_repair_success = False
            if generation_error is None and force_citations and citation_repair_retry:
                parsed_initial = parse_citations(answer_text)
                if (
                    not _is_not_found_answer(answer_text)
                    and len(parsed_initial) < citation_min_count
                ):
                    citation_repair_applied = True
                    try:
                        repair_messages = build_citation_repair_messages(
                            query=query_text,
                            context_text=context_text,
                            original_answer=answer_text,
                            citation_min_count=citation_min_count,
                            allowed_citation_ids=(
                                build_force_citation_allowlist(citations) if force_citations else None
                            ),
                        )
                        repaired_answer_text, usage = generate_answer(
                            provider=provider,
                            messages=repair_messages,
                            max_tokens=int(gen_cfg.get("max_tokens", 512)),
                            temperature=float(gen_cfg.get("temperature", 0.2)),
                            ollama_num_predict=int(gen_cfg.get("ollama_num_predict", 1536)),
                            ollama_retry_num_predict=int(gen_cfg.get("ollama_retry_num_predict", 2048)),
                            max_context_chars=int(gen_cfg.get("max_context_chars", max_context_chars)),
                            ollama_num_predict_auto=bool(gen_cfg.get("ollama_num_predict_auto", True)),
                            ollama_think=gen_cfg.get("ollama_think", False),
                            fallback_citations=citations,
                            ollama_debug_hook=hook,
                        )
                        repaired_text = str(repaired_answer_text or "").strip()
                        if repaired_text:
                            answer_text = repaired_text
                        citation_repair_success = len(parse_citations(answer_text)) >= citation_min_count
                    except Exception:
                        citation_repair_success = False
            progress_reporter.phase(query_id, "write_artifacts")
            if generation_error is not None:
                _write_provider_debug(
                    per_query_dir=per_query_dir,
                    query_prefix=per_query_prefix,
                    query_id=query_id,
                    issue="generation_exception",
                    usage=usage,
                    error=generation_error,
                )
            elif not answer_text:
                _write_provider_debug(
                    per_query_dir=per_query_dir,
                    query_prefix=per_query_prefix,
                    query_id=query_id,
                    issue="empty_answer",
                    usage=usage,
                    error=None,
                )
            if backend_name == "ollama" and (generation_error is not None or not answer_text):
                write_ollama_attempt_artifacts(
                    per_query_dir=per_query_dir,
                    query_prefix=per_query_prefix,
                    attempts=ollama_attempts,
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
            answer_len_chars = len(answer_text)
            citation_count = len(citation_rows)
            citation_rate = _citation_rate(answer_text)
            parsed_model_citations = parse_citations(answer_text)
            model_citation_count = len(parsed_model_citations)
            citation_resolution = resolve_citations(parsed_model_citations, citations)
            verification = verify_answer(
                answer_text=answer_text,
                retrieved_rows=retrieved,
                citation_report=citation_resolution,
            )
            verification_status = str(verification.get("verification_status") or "")
            verification_reason = str(verification.get("short_reason") or "")
            if (
                force_citations
                and int(citation_resolution.get("citations_total") or 0) > 0
                and int(citation_resolution.get("citations_resolved") or 0) == 0
            ):
                verification_reason = "All cited IDs were not in allowlist / not retrieved."
            cited_pages = sorted(
                int(page)
                for page in (citation_resolution.get("resolved_pages") or set())
                if isinstance(page, int)
            )
            cited_ids = [
                str(value)
                for value in (citation_resolution.get("resolved_ids") or [])
                if str(value).strip()
            ]
            status = "ok"
            failure_reason: Optional[str] = None
            if generation_error is not None:
                status = "failed"
                failure_reason = str(generation_error)
                failed_queries += 1
            elif not answer_text:
                status = "failed"
                failure_reason = "Empty answer from provider"
                failed_queries += 1

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
                "answer_text": answer_text,
                "answer": answer_text,
                "answer_len_chars": answer_len_chars,
                "citation_rate": round(citation_rate, 4),
                "citation_count": citation_count,
                "citations": citation_rows,
                "model_citations_total": int(citation_resolution.get("citations_total") or 0),
                "citations_total": int(citation_resolution.get("citations_total") or 0),
                "citations_resolved": int(citation_resolution.get("citations_resolved") or 0),
                "citations_resolvable": bool(citation_resolution.get("citations_resolvable", False)),
                "cited_pages": cited_pages,
                "cited_ids": cited_ids,
                "first_cited_rank": citation_resolution.get("first_cited_rank"),
                "verification_status": verification_status,
                "verification_reason": verification_reason,
                "citation_repair_applied": bool(citation_repair_applied),
                "citation_repair_success": bool(citation_repair_success),
                "retrieved_ids": retrieved_ids,
                "usage": usage,
                "status": status,
                "failure_reason": failure_reason,
            }
            rows.append(row)
            _write_json(items_dir / f"{query_id}.json", row)
            answer_file_text = (
                answer_text
                if answer_text
                else f"GENERATION_FAILED: {failure_reason or 'Unknown generation failure'}"
            )
            (per_query_dir / f"{per_query_prefix}_answer.txt").write_text(
                answer_file_text + "\n",
                encoding="utf-8",
            )
            _write_json(
                per_query_dir / f"{per_query_prefix}_citations.json",
                {
                    "id": query_id,
                    "query": query_text,
                    "citations": citation_rows,
                    "citations_total": int(citation_resolution.get("citations_total") or 0),
                    "citations_resolved": int(citation_resolution.get("citations_resolved") or 0),
                    "citations_resolvable": bool(citation_resolution.get("citations_resolvable", False)),
                    "cited_pages": cited_pages,
                    "cited_ids": cited_ids,
                    "first_cited_rank": citation_resolution.get("first_cited_rank"),
                    "verification_status": verification_status,
                    "verification_reason": verification_reason,
                    "citation_repair_applied": bool(citation_repair_applied),
                    "citation_repair_success": bool(citation_repair_success),
                    "retrieved_ids": retrieved_ids,
                },
            )

            total_values.append(total_ms)
            answer_len_values.append(answer_len_chars)
            context_citation_count_values.append(citation_count)
            model_citation_count_values.append(model_citation_count)
            if verification_status == PASS:
                verification_passes += 1
            progress_reporter.next_query()
    finally:
        progress_reporter.close()

    count_queries = len(rows)
    summary: Dict[str, Any] = {
        "count_queries": count_queries,
        "avg_total_ms": round(sum(total_values) / count_queries, 3) if count_queries else 0.0,
        "p50_total_ms": round(float(median(total_values)), 3) if total_values else 0.0,
        "avg_answer_len_chars": round(sum(answer_len_values) / count_queries, 3)
        if count_queries
        else 0.0,
        "avg_context_citations_per_query": round(sum(context_citation_count_values) / count_queries, 3)
        if count_queries
        else 0.0,
        "avg_model_citations_per_query": round(sum(model_citation_count_values) / count_queries, 3)
        if count_queries
        else 0.0,
        # Backward-compatible alias; historical consumers may still read this key.
        "avg_num_citations": round(sum(context_citation_count_values) / count_queries, 3)
        if count_queries
        else 0.0,
        "percent_queries_with_any_context_citation": round(
            (any_citation_count / count_queries) * 100.0, 3
        )
        if count_queries
        else 0.0,
        "percent_queries_with_any_citation": round((any_citation_count / count_queries) * 100.0, 3)
        if count_queries
        else 0.0,
        "failed_queries": failed_queries,
        "verification_pass_rate": round(verification_passes / count_queries, 4)
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
                "ollama_api": str(
                    gen_cfg.get("ollama_api") or backend_cfg.get("ollama_api") or "generate"
                ),
                "ollama_num_predict": int(gen_cfg.get("ollama_num_predict", 1536)),
                "ollama_retry_num_predict": int(gen_cfg.get("ollama_retry_num_predict", 2048)),
                "ollama_num_predict_auto": bool(gen_cfg.get("ollama_num_predict_auto", True)),
                "max_context_chars": int(gen_cfg.get("max_context_chars", max_context_chars)),
                "ollama_think": gen_cfg.get("ollama_think", False),
                "force_citations": force_citations,
                "citation_repair_retry": citation_repair_retry,
                "citation_min_count": citation_min_count,
            },
            "force_citations": force_citations,
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
        "avg_context_citations_per_query",
        "avg_model_citations_per_query",
        "avg_num_citations",
        "percent_queries_with_any_context_citation",
        "percent_queries_with_any_citation",
        "failed_queries",
        "verification_pass_rate",
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
            f"answer_len={row.get('answer_len_chars')} status={row.get('status')}"
        )
        if row.get("failure_reason"):
            lines.append(f"  failure_reason: {row.get('failure_reason')}")
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
    queries = results_dict.get("queries")
    summary = results_dict.get("summary")
    if isinstance(queries, list):
        answer_lengths: List[int] = []
        for row in queries:
            if not isinstance(row, dict):
                continue
            answer_text = str(row.get("answer_text") or row.get("answer") or "").strip()
            answer_len = len(answer_text)
            row["answer_text"] = answer_text
            row["answer_len_chars"] = answer_len
            answer_lengths.append(answer_len)
        if isinstance(summary, dict):
            count_queries = int(summary.get("count_queries") or 0)
            if count_queries > 0 and answer_lengths:
                summary["avg_answer_len_chars"] = round(sum(answer_lengths) / count_queries, 3)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"
    _write_json(report_json, results_dict)
    report_md.write_text("\n".join(_report_lines(results_dict)), encoding="utf-8")
