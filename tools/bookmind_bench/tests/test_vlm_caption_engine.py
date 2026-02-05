from engines.vlm_caption_engine import OcrBlock, _select_ocr_hints


def test_select_ocr_hints_intersection() -> None:
    figure_bbox = [0, 0, 10, 10]
    blocks = [
        OcrBlock(bbox=[2, 2, 4, 4], text="Voltage"),
        OcrBlock(bbox=[20, 20, 30, 30], text="Ignore"),
    ]

    hints, used = _select_ocr_hints(figure_bbox, blocks, iou_threshold=0.01)

    assert used is True
    assert "Voltage" in hints
    assert "Ignore" not in hints
