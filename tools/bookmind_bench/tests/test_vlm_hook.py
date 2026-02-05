from pathlib import Path

import json

from engines.vlm_hook import plan_vlm_jobs


def test_plan_vlm_jobs_for_figures(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "page_0001.png").write_bytes(b"")

    records_path = tmp_path / "paddleocr.jsonl"
    records_path.write_text(
        "\n".join(
            [
                '{"engine":"paddleocr","page":1,"type":"layout_block","text":"","bbox":[1,2,3,4],"timing_ms":1,"meta":{"pdf_page_start":1,"pdf_page_end":1,"block_type":"figure"}}',
                '{"engine":"paddleocr","page":1,"type":"ocr_text","text":"hello","bbox":[1,2,3,4],"timing_ms":1,"meta":{"pdf_page_start":1,"pdf_page_end":1,"block_type":"text"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_jobs_path = tmp_path / "vlm_jobs.jsonl"
    job_count = plan_vlm_jobs(
        records_jsonl_path=str(records_path),
        imgdir=str(img_dir),
        out_jobs_path=str(out_jobs_path),
    )

    assert job_count == 1
    jobs = []
    with out_jobs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                jobs.append(json.loads(line))
    assert len(jobs) == 1
    job = jobs[0]
    assert job["page"] == 1
    assert job["bbox"] == [1, 2, 3, 4]
    assert job["image_path"].endswith("page_0001.png")
    assert job["suggested_crop_path"].endswith("page_0001_block_001.png")
    assert job["prompt_template"] == "technical_diagram_v1"
