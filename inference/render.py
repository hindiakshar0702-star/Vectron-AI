"""
Render SVG strings to PNG previews.

Used for evaluation, API responses, and Colab quick-view of generated icons.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import cairosvg
from PIL import Image


log = logging.getLogger(__name__)


def svg_to_png_bytes(
    svg: str,
    *,
    width: int = 256,
    height: int = 256,
    background: str | None = "white",
) -> bytes:
    """Rasterize SVG string to PNG bytes."""
    return cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=width,
        output_height=height,
        background_color=background,
    )


def svg_to_pil(svg: str, *, width: int = 256, height: int = 256) -> Image.Image:
    return Image.open(io.BytesIO(svg_to_png_bytes(svg, width=width, height=height))).convert("RGBA")


def make_preview_grid(svgs: list[str], cols: int = 4, tile: int = 128) -> Image.Image:
    """Combine multiple SVGs into a single grid PNG (handy for variant displays)."""
    rows = (len(svgs) + cols - 1) // cols
    canvas = Image.new("RGBA", (cols * tile, rows * tile), (255, 255, 255, 0))
    for i, svg in enumerate(svgs):
        try:
            tile_img = svg_to_pil(svg, width=tile, height=tile)
        except Exception as e:  # noqa: BLE001
            log.warning("render failed for variant %d: %s", i, e)
            continue
        x = (i % cols) * tile
        y = (i // cols) * tile
        canvas.paste(tile_img, (x, y), tile_img)
    return canvas


def save_previews(svg: str, base_path: Path, sizes: tuple[int, ...] = (64, 128, 256)) -> list[Path]:
    """Write one PNG per requested size next to base_path."""
    base_path = Path(base_path)
    written: list[Path] = []
    for s in sizes:
        out = base_path.with_name(f"{base_path.stem}@{s}.png")
        out.write_bytes(svg_to_png_bytes(svg, width=s, height=s))
        written.append(out)
    return written
