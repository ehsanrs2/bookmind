import sys
import types

from engines import paddleocr_engine


def test_paddleocr_fallback_fullpage_figure(monkeypatch, tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    image_path = img_dir / "page_1.png"
    image_path.write_bytes(b"fake")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "dummy").write_text("x")

    monkeypatch.setattr(paddleocr_engine, "_page_area", lambda _: 1000.0)

    class FakePPStructure:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, image_path_str):
            return [{"type": "figure", "bbox": [0, 0, 100, 9]}]

    class FakePaddleOCR:
        def __init__(self, *args, **kwargs):
            pass

        def ocr(self, image_path_str, cls=False):
            return [
                [[[0, 0], [50, 0], [50, 10], [0, 10]], ("Hello", 0.9)],
                [[[0, 20], [50, 20], [50, 30], [0, 30]], ("World", 0.8)],
            ]

    fake_module = types.SimpleNamespace(PPStructure=FakePPStructure, PaddleOCR=FakePaddleOCR)
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    records = paddleocr_engine.run_paddleocr(
        img_dir=str(img_dir),
        pages=None,
        lang="en",
        use_gpu=False,
        cache_dir=str(cache_dir),
    )

    figure_blocks = [
        record
        for record in records
        if record["type"] == "layout_block"
        and record["meta"].get("block_type") == "figure"
    ]
    ocr_blocks = [record for record in records if record["type"] == "ocr_text"]

    assert len(figure_blocks) == 1
    assert len(ocr_blocks) >= 2
    for record in ocr_blocks:
        assert record.get("text")
        assert record.get("bbox")
