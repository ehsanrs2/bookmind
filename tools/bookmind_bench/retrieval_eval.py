"""Retrieval-only evaluation for Bookmind bench queries."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qdrant_ingest import DEFAULT_EMBED_ALIAS, search_query


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


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return 0.0


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def expand_truth_pages(query_item: Dict[str, Any]) -> List[int]:
    pages: set[int] = set()

    raw_pages = query_item.get("truth_pages")
    if isinstance(raw_pages, list):
        for value in raw_pages:
            page = _to_int(value)
            if page is not None and page > 0:
                pages.add(page)

    raw_ranges = query_item.get("truth_page_ranges")
    if isinstance(raw_ranges, list):
        for pair in raw_ranges:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            start = _to_int(pair[0])
            end = _to_int(pair[1])
            if start is None or end is None:
                continue
            lo, hi = (start, end) if start <= end else (end, start)
            for page in range(lo, hi + 1):
                if page > 0:
                    pages.add(page)

    return sorted(pages)


def _normalize_retrieved_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "rank": rank,
                "score": _to_float(row.get("score")),
                "page": _to_int(row.get("page")),
                "stable_id": str(row.get("stable_id") or row.get("id") or ""),
                "chunk_id": str(row.get("chunk_id") or ""),
                "content_type": str(row.get("content_type") or ""),
            }
        )
    return out


def compute_page_metrics(
    retrieved_rows: Sequence[Dict[str, Any]],
    truth_pages: Sequence[int],
) -> Dict[str, Any]:
    truth = {int(page) for page in truth_pages if isinstance(page, int) and page > 0}
    first_hit_rank: Optional[int] = None
    retrieved_truth: set[int] = set()

    for row in retrieved_rows:
        if not isinstance(row, dict):
            continue
        rank = _to_int(row.get("rank"))
        page = _to_int(row.get("page"))
        if page is None or page not in truth:
            continue
        retrieved_truth.add(page)
        if first_hit_rank is None and rank is not None and rank > 0:
            first_hit_rank = rank

    page_hit = first_hit_rank is not None
    page_recall = (len(retrieved_truth) / len(truth)) if truth else 0.0
    mrr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0

    return {
        "page_hit": bool(page_hit),
        "page_recall": round(page_recall, 6),
        "first_hit_rank": first_hit_rank,
        "mrr": round(mrr, 6),
    }


def _pick_row_id(row: Dict[str, Any]) -> str:
    chunk_id = str(row.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id
    return str(row.get("stable_id") or "").strip()


def jaccard_similarity(values_a: Sequence[str], values_b: Sequence[str]) -> float:
    set_a = {str(value) for value in values_a if str(value).strip()}
    set_b = {str(value) for value in values_b if str(value).strip()}
    if not set_a and not set_b:
        return 1.0
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(set_a.intersection(set_b)) / len(union)


def average_pairwise_jaccard(id_runs: Sequence[Sequence[str]]) -> Optional[float]:
    if len(id_runs) < 2:
        return None
    values: List[float] = []
    for idx_a, idx_b in combinations(range(len(id_runs)), 2):
        values.append(jaccard_similarity(id_runs[idx_a], id_runs[idx_b]))
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def run_retrieval_eval(
    *,
    queries: Sequence[Dict[str, Any]],
    qdrant_url: str,
    collection: str,
    top_k: int,
    content_types: Sequence[str],
    output_dir: str | Path,
    queries_path: Optional[str] = None,
    repeat: int = 1,
    seed: Optional[int] = None,
    progress: bool = False,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    per_query_dir = out_dir / "per_query"
    per_query_dir.mkdir(parents=True, exist_ok=True)

    repeat_runs = max(1, int(repeat))
    if seed is not None:
        random.seed(int(seed))

    hit_values: List[int] = []
    recall_values: List[float] = []
    mrr_values: List[float] = []
    stability_topk_values: List[float] = []
    stability_top5_values: List[float] = []
    breakdown_counts: Dict[str, Dict[str, int]] = {}

    query_rows: List[Dict[str, Any]] = []
    progress_reporter = _build_progress_reporter(len(queries), enabled=bool(progress))

    try:
        for index, query_item in enumerate(queries, start=1):
            query_id = str(query_item.get("id") or f"q{index}")
            query_text = str(query_item.get("query") or "")
            truth_pages = expand_truth_pages(query_item)

            run_retrieved: List[List[Dict[str, Any]]] = []
            for run_idx in range(repeat_runs):
                progress_reporter.phase(query_id, f"retrieve run {run_idx + 1}/{repeat_runs}")
                retrieved_rows = search_query(
                    query=query_text,
                    qdrant_url=qdrant_url,
                    collection=collection,
                    embed_model=DEFAULT_EMBED_ALIAS,
                    top_k=top_k,
                    content_types=list(content_types) if content_types else None,
                )
                run_retrieved.append(_normalize_retrieved_rows(retrieved_rows))

            primary_retrieved = run_retrieved[0] if run_retrieved else []
            metrics = compute_page_metrics(primary_retrieved, truth_pages)

            hit_values.append(1 if metrics["page_hit"] else 0)
            recall_values.append(float(metrics["page_recall"]))
            mrr_values.append(float(metrics["mrr"]))

            truth_set = set(truth_pages)
            for row in primary_retrieved:
                content_type = str(row.get("content_type") or "unknown").strip() or "unknown"
                group = breakdown_counts.setdefault(
                    content_type,
                    {"retrieved_count": 0, "truth_page_row_hits": 0},
                )
                group["retrieved_count"] += 1
                page = _to_int(row.get("page"))
                if page is not None and page in truth_set:
                    group["truth_page_row_hits"] += 1

            query_payload: Dict[str, Any] = {
                "id": query_id,
                "query": query_text,
                "truth_pages_expanded": truth_pages,
                "retrieved": primary_retrieved,
                "metrics": metrics,
            }

            if repeat_runs > 1:
                topk_ids_by_run = [
                    [_pick_row_id(row) for row in rows[:top_k] if _pick_row_id(row)]
                    for rows in run_retrieved
                ]
                top5_ids_by_run = [
                    [_pick_row_id(row) for row in rows[:5] if _pick_row_id(row)]
                    for rows in run_retrieved
                ]
                avg_jaccard_topk = average_pairwise_jaccard(topk_ids_by_run)
                avg_jaccard_top5 = average_pairwise_jaccard(top5_ids_by_run)
                if avg_jaccard_topk is not None:
                    stability_topk_values.append(avg_jaccard_topk)
                if avg_jaccard_top5 is not None:
                    stability_top5_values.append(avg_jaccard_top5)
                query_payload["stability"] = {
                    "top_k_ids_by_run": topk_ids_by_run,
                    "top5_ids_by_run": top5_ids_by_run,
                    "avg_jaccard_topk": avg_jaccard_topk,
                    "avg_jaccard_top5": avg_jaccard_top5,
                }

            query_rows.append(query_payload)
            _write_json(per_query_dir / f"q{index:02d}_retrieval.json", query_payload)
            progress_reporter.next_query()
    finally:
        progress_reporter.close()

    total_queries = len(query_rows)
    total_retrieved_rows = sum(group["retrieved_count"] for group in breakdown_counts.values())
    breakdown: Dict[str, Dict[str, Any]] = {}
    for content_type, group in sorted(breakdown_counts.items()):
        retrieved_count = int(group.get("retrieved_count") or 0)
        truth_hits = int(group.get("truth_page_row_hits") or 0)
        breakdown[content_type] = {
            "retrieved_count": retrieved_count,
            "retrieved_fraction": round(
                (retrieved_count / total_retrieved_rows), 6
            )
            if total_retrieved_rows
            else 0.0,
            "truth_page_row_hits": truth_hits,
        }

    report: Dict[str, Any] = {
        "config": {
            "qdrant_url": qdrant_url,
            "collection": collection,
            "top_k": int(top_k),
            "content_types": list(content_types),
            "repeat": repeat_runs,
            "queries_path": str(queries_path) if queries_path is not None else None,
            "seed": seed,
        },
        "summary": {
            "count_queries": total_queries,
            "hit_rate@k": round((sum(hit_values) / total_queries), 6) if total_queries else 0.0,
            "avg_page_recall@k": round((sum(recall_values) / total_queries), 6)
            if total_queries
            else 0.0,
            "mrr": round((sum(mrr_values) / total_queries), 6) if total_queries else 0.0,
        },
        "breakdown": {
            "by_content_type": breakdown,
        },
        "queries": query_rows,
    }

    if repeat_runs > 1:
        report["stability"] = {
            "avg_jaccard_topk": round(sum(stability_topk_values) / len(stability_topk_values), 6)
            if stability_topk_values
            else 0.0,
            "avg_jaccard_top5": round(sum(stability_top5_values) / len(stability_top5_values), 6)
            if stability_top5_values
            else 0.0,
        }

    _write_json(out_dir / "retrieval_report.json", report)
    return report
