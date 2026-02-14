from pathlib import Path


def test_detectron2_build_script_has_standalone_python_header_fallback():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_detectron2_wheel.sh"
    ).read_text(encoding="utf-8")
    assert 'Path(sys.executable).resolve().parents[1] / "include"' in script
    assert "using repository HEAD" in script
    assert "--no-build-isolation" in script
    assert "export CXX=g++" in script
