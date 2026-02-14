from eval_pack import reorder_results_for_query


def test_reorder_results_for_diagram_query_boosts_figure_caption() -> None:
    rows = [
        {"stable_id": "t1", "content_type": "text"},
        {"stable_id": "f1", "content_type": "figure_caption"},
        {"stable_id": "t2", "content_type": "text"},
        {"stable_id": "f2", "content_type": "figure_caption"},
    ]

    reordered = reorder_results_for_query("explain the block diagram", rows, True)
    assert [row["stable_id"] for row in reordered] == ["f1", "f2", "t1", "t2"]


def test_reorder_results_can_disable_heuristic() -> None:
    rows = [
        {"stable_id": "t1", "content_type": "text"},
        {"stable_id": "f1", "content_type": "figure_caption"},
    ]

    reordered = reorder_results_for_query("explain the diagram", rows, False)
    assert [row["stable_id"] for row in reordered] == ["t1", "f1"]
