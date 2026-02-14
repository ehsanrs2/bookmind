import json
from pathlib import Path

from bundle import bundle_run


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_run_dir(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    crop_path = run_dir / "vlm" / "crops" / "crop1.png"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.write_bytes(b"fake-image")

    records = [
        {
            "stable_id": "fig-1",
            "page": 1,
            "content_type": "figure_caption",
            "text": "A short figure caption",
            "bbox": [10, 10, 90, 90],
            "meta": {
                "figure_ref": {
                    "layout_region_id": "page_0001_region_001",
                    "layout_line_id": 1,
                    "crop_image": str(crop_path),
                },
                "figure_context": "Nearby text context",
            },
        },
        {
            "stable_id": "txt-1",
            "page": 1,
            "content_type": "text",
            "text": "First page paragraph",
            "bbox": [1, 1, 20, 20],
            "meta": {},
        },
        {
            "stable_id": "txt-2",
            "page": 2,
            "content_type": "text",
            "text": "Second page paragraph for median checks",
            "bbox": [1, 1, 20, 20],
            "meta": {},
        },
    ]
    _write_jsonl(run_dir / "ingest" / "records.jsonl", records)
    return run_dir, crop_path


def test_bundle_run_include_images_true(tmp_path: Path) -> None:
    run_dir, crop_path = _build_run_dir(tmp_path)

    result = bundle_run(run_dir=run_dir, include_images=True)

    bundle_dir = run_dir / "bundle"
    assert result.bundle_dir == bundle_dir
    assert (bundle_dir / "records.jsonl").exists()
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "report.json").exists()
    assert (bundle_dir / "report.md").exists()

    manifest = _read_json(bundle_dir / "manifest.json")
    assert manifest["records"]["count"] == 3
    assert manifest["records"]["content_types"]["figure_caption"] == 1
    assert manifest["records"]["content_types"]["text"] == 2
    assert manifest["images"]["included"] is True
    assert manifest["images"]["count"] == 1

    rows = _read_jsonl(bundle_dir / "records.jsonl")
    figure = [row for row in rows if row["content_type"] == "figure_caption"][0]
    crop_ref = figure["meta"]["figure_ref"]["crop_image"]
    assert crop_ref.startswith("images/crops/")
    assert (bundle_dir / crop_ref).exists()
    assert (bundle_dir / crop_ref).read_bytes() == crop_path.read_bytes()

    report = _read_json(bundle_dir / "report.json")
    assert "total_records" in report
    assert "counts_by_content_type" in report
    assert "figure_caption" in report
    assert "text" in report
    assert "per_page" in report
    assert report["total_records"] == 3
    assert report["figure_caption"]["captions_total"] == 1
    assert report["figure_caption"]["with_crop_image"] == 1


def test_bundle_run_include_images_false(tmp_path: Path) -> None:
    run_dir, crop_path = _build_run_dir(tmp_path)
    del crop_path

    bundle_run(run_dir=run_dir, include_images=False)

    bundle_dir = run_dir / "bundle"
    assert (bundle_dir / "records.jsonl").exists()
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "report.json").exists()
    assert not (bundle_dir / "images").exists()
