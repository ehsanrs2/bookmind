import pytest

from engines.paddleocr_engine import validate_paddleocr_record


def test_validate_paddleocr_record_ok():
    record = {
        "engine": "paddleocr",
        "page": 1,
        "type": "ocr_text",
        "text": "hello",
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "timing_ms": 12,
        "meta": {"pdf_page_start": 1, "pdf_page_end": 1, "block_type": "text"},
    }
    validate_paddleocr_record(record)


def test_validate_paddleocr_record_missing_meta():
    record = {
        "engine": "paddleocr",
        "page": 1,
        "type": "ocr_text",
        "text": "hello",
        "timing_ms": 12,
    }
    with pytest.raises(ValueError):
        validate_paddleocr_record(record)
