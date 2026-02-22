from __future__ import annotations

import json
from pathlib import Path

import retrieval_eval


def test_expand_truth_pages_unions_ranges_and_points() -> None:
    query = {
        "truth_pages": [12, "13", 0],
        "truth_page_ranges": [[44, 45], [50, 48], ["x", 2]],
    }
    assert retrieval_eval.expand_truth_pages(query) == [12, 13, 44, 45, 48, 49, 50]


def test_compute_page_metrics_hit_recall_mrr() -> None:
    retrieved = [
        {"rank": 1, "page": 3},
        {"rank": 2, "page": 8},
        {"rank": 3, "page": 9},
    ]
    metrics = retrieval_eval.compute_page_metrics(retrieved, [8, 9, 11])
    assert metrics["page_hit"] is True
    assert metrics["first_hit_rank"] == 2
    assert metrics["page_recall"] == 0.666667
    assert metrics["mrr"] == 0.5


def test_average_pairwise_jaccard() -> None:
    runs = [
        ["a", "b", "c"],
        ["b", "c", "d"],
        ["b", "d"],
    ]
    # pairwise: (0,1)=0.5 (0,2)=0.25 (1,2)=0.6666667 => avg=0.472222...
    assert retrieval_eval.average_pairwise_jaccard(runs) == 0.472222


def test_run_retrieval_eval_writes_report_and_per_query(tmp_path: Path, monkeypatch) -> None:
    queries = [
        {"id": "q1", "query": "Q1", "truth_pages": [2]},
        {"id": "q2", "query": "Q2", "truth_page_ranges": [[5, 6]]},
    ]

    call_index = {"q1": 0, "q2": 0}
    q1_runs = [
        [
            {"score": 0.9, "page": 2, "stable_id": "s1", "chunk_id": "v1_c1", "content_type": "text"},
            {"score": 0.7, "page": 4, "stable_id": "s2", "chunk_id": "v1_c2", "content_type": "figure_caption"},
        ],
        [
            {"score": 0.88, "page": 2, "stable_id": "s1", "chunk_id": "v1_c1", "content_type": "text"},
            {"score": 0.6, "page": 7, "stable_id": "s3", "chunk_id": "v1_c3", "content_type": "text"},
        ],
    ]
    q2_runs = [
        [
            {"score": 0.9, "page": 1, "stable_id": "x1", "chunk_id": "v1_x1", "content_type": "text"},
            {"score": 0.8, "page": 6, "stable_id": "x2", "chunk_id": "v1_x2", "content_type": "figure_caption"},
        ],
        [
            {"score": 0.91, "page": 5, "stable_id": "x3", "chunk_id": "v1_x3", "content_type": "figure_caption"},
            {"score": 0.75, "page": 6, "stable_id": "x2", "chunk_id": "v1_x2", "content_type": "figure_caption"},
        ],
    ]

    def _fake_search_query(**kwargs):
        query = kwargs["query"]
        if query == "Q1":
            idx = call_index["q1"]
            call_index["q1"] += 1
            return q1_runs[idx]
        idx = call_index["q2"]
        call_index["q2"] += 1
        return q2_runs[idx]

    monkeypatch.setattr(retrieval_eval, "search_query", _fake_search_query)

    report = retrieval_eval.run_retrieval_eval(
        queries=queries,
        qdrant_url="http://127.0.0.1:6333",
        collection="bookmind_bench",
        top_k=20,
        content_types=["text", "figure_caption"],
        output_dir=tmp_path,
        queries_path=str(tmp_path / "queries.json"),
        repeat=2,
        progress=False,
    )

    assert report["summary"]["count_queries"] == 2
    assert report["summary"]["hit_rate@k"] == 1.0
    assert report["summary"]["avg_page_recall@k"] == 0.75
    assert report["summary"]["mrr"] == 0.75
    assert "stability" in report
    assert (tmp_path / "retrieval_report.json").exists()
    assert (tmp_path / "per_query" / "q01_retrieval.json").exists()
    assert (tmp_path / "per_query" / "q02_retrieval.json").exists()

    q1_payload = json.loads((tmp_path / "per_query" / "q01_retrieval.json").read_text(encoding="utf-8"))
    assert q1_payload["truth_pages_expanded"] == [2]
    assert q1_payload["retrieved"][0]["chunk_id"] == "v1_c1"
    assert q1_payload["metrics"]["page_hit"] is True
    assert q1_payload["stability"]["avg_jaccard_topk"] == 0.333333
