from pathlib import Path

from PIL import Image

from engines.crop_utils import crop_image


def test_crop_image_clamps_and_writes_png(tmp_path: Path) -> None:
    image_path = tmp_path / "src.png"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(image_path)

    out_path = tmp_path / "out" / "crop.png"
    crop_image(str(image_path), bbox=[-5, -5, 6, 6], out_path=str(out_path), pad_px=2)

    assert out_path.exists()
    with Image.open(out_path) as cropped:
        assert cropped.size == (8, 8)
