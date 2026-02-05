"""Image crop utilities for VLM figure captioning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from PIL import Image


def _normalize_bbox(bbox: Iterable[float]) -> List[float]:
    values = list(bbox)
    if len(values) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(values)}")
    return values


def crop_image(image_path: str, bbox: Iterable[float], out_path: str, pad_px: int = 12) -> str:
    """Crop an image to bbox (with padding) and write a PNG to out_path."""
    bbox_values = _normalize_bbox(bbox)
    x0, y0, x1, y1 = bbox_values
    left = min(x0, x1)
    right = max(x0, x1)
    top = min(y0, y1)
    bottom = max(y0, y1)

    image_path = str(image_path)
    out_path = str(out_path)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        width, height = image.size
        left = max(0, int(round(left - pad_px)))
        top = max(0, int(round(top - pad_px)))
        right = min(width, int(round(right + pad_px)))
        bottom = min(height, int(round(bottom + pad_px)))

        if right <= left or bottom <= top:
            raise ValueError("Crop bbox collapsed after clamping")

        cropped = image.crop((left, top, right, bottom))
        cropped.save(out_path, format="PNG")

    return out_path
