#!/usr/bin/env python3
"""Simple smoke test for a local OpenAI-compatible VLM endpoint."""

from __future__ import annotations

import argparse
import base64
import os
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image


def _encode_image(path: Path, max_side: int | None) -> str:
    if max_side is None or max_side <= 0:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"

    with Image.open(path) as img:
        width, height = img.size
        if max(width, height) > max_side:
            scale = max_side / float(max(width, height))
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            img = img.resize(new_size, Image.LANCZOS)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        data = buffer.getvalue()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main() -> int:
    parser = argparse.ArgumentParser(description="VLM endpoint smoke test")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible endpoint (default: http://127.0.0.1:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BOOKMIND_VLM_MODEL", "qwen3-vl"),
        help="Model identifier (default: env BOOKMIND_VLM_MODEL or qwen3-vl)",
    )
    parser.add_argument("--image", required=True, help="Path to a local PNG image")
    parser.add_argument(
        "--max_side",
        type=int,
        default=1024,
        help="Resize image to this max side length (default: 1024, set 0 to disable)",
    )
    parser.add_argument(
        "--detail",
        choices=("low", "high", "auto"),
        default="low",
        help="Image detail hint for compatible servers (default: low)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    prompt = "Describe the diagram: list components and connections."
    image_url = _encode_image(image_path, args.max_side)

    client = OpenAI(base_url=args.endpoint, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": args.detail}},
                ],
            }
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    content = response.choices[0].message.content
    if not content:
        raise SystemExit("Empty response from VLM endpoint.")

    print(content.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
