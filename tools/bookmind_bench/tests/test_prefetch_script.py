from pathlib import Path


def test_prefetch_installs_detectron2_without_deps_from_local_wheelhouse():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "prefetch_online.sh").read_text(
        encoding="utf-8"
    )
    assert "--no-index --find-links \"$LAYOUT_WHEEL_DIR\" --no-deps detectron2" in script
