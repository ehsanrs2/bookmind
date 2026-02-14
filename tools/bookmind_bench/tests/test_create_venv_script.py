from pathlib import Path


def test_layout_python_minor_check_uses_valid_fstring():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "create_venv.sh").read_text(
        encoding="utf-8"
    )
    assert 'print(f\\"{sys.version_info.major}.{sys.version_info.minor}\\")' not in script
    assert 'print(f"{sys.version_info.major}.{sys.version_info.minor}")' in script


def test_nvidia_smi_probe_is_non_fatal():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "create_venv.sh").read_text(
        encoding="utf-8"
    )
    assert "nvidia-smi --query-gpu=cuda_version --format=csv,noheader" in script
    assert "| head -n 1 | tr -d '[:space:]' || true" in script
