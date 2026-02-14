from engines.vlm_caption_engine import OcrBlock, _fit_prompt_to_limit, _select_ocr_hints


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


def test_prompt_length_cap_truncates_hints() -> None:
    long_hint = "A" * 12000
    prompt = _fit_prompt_to_limit("technical_diagram_v1", long_hint, max_prompt_chars=6000)
    assert len(prompt) <= 6000
    assert "OCR hints:" in prompt
