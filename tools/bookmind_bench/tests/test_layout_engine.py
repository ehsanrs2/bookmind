import pytest

from engines.layout_engine import (
    clamp_bbox_to_image,
    map_crop_bbox_to_page,
    validate_layout_record,
)


def test_clamp_bbox_to_image_clamps_to_bounds():
    bbox = clamp_bbox_to_image([-10, 5, 120, 50], image_width=100, image_height=80)
    assert bbox == [0.0, 5.0, 100.0, 50.0]


def test_clamp_bbox_to_image_returns_none_for_collapsed_box():
    bbox = clamp_bbox_to_image([15, 15, 15, 80], image_width=100, image_height=100)
    assert bbox is None


def test_map_crop_bbox_to_page_maps_offsets_correctly():
    page_bbox = map_crop_bbox_to_page(
        crop_bbox=[10, 15, 30, 35],
        region_bbox=[100, 200, 300, 400],
        page_width=1000,
        page_height=1000,
    )
    assert page_bbox == [110.0, 215.0, 130.0, 235.0]


def test_validate_layout_record_schema():
    record = {
        "engine": "layoutparser",
        "page": 1,
        "type": "ocr_text",
        "text": "line",
        "bbox": [10.0, 11.0, 20.0, 30.0],
        "timing_ms": 4,
        "meta": {
            "pdf_page_start": 1,
            "pdf_page_end": 1,
            "block_type": "text",
            "trace": {"region_id": "page_0001_region_001"},
        },
    }
    validate_layout_record(record)


def test_validate_layout_record_missing_trace_for_ocr_text():
    record = {
        "engine": "layoutparser",
        "page": 1,
        "type": "ocr_text",
        "text": "line",
        "bbox": [10.0, 11.0, 20.0, 30.0],
        "timing_ms": 4,
        "meta": {
            "pdf_page_start": 1,
            "pdf_page_end": 1,
            "block_type": "text",
        },
    }
    with pytest.raises(ValueError):
        validate_layout_record(record)
