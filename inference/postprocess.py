"""
Validate, repair, and optimize SVG output produced by the model.

Pipeline:
1. Extract <svg>...</svg> block from the model's free-form response.
2. Parse with lxml; if it fails, attempt a small auto-repair pass
   (close unclosed tags, strip non-svg chunks).
3. Enforce viewBox, xmlns, width/height.
4. Optionally snap path coordinates to a grid.
5. Optionally run svgo (Node binary) - falls back gracefully if absent.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lxml import etree


SVG_NS = "http://www.w3.org/2000/svg"
SVG_BLOCK_RE = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)
NUM_RE = re.compile(r"-?\d+\.\d+")

log = logging.getLogger(__name__)


@dataclass
class PostprocessResult:
    svg: str | None
    valid: bool
    issues: list[str]


def extract_svg_block(text: str) -> str | None:
    """Pull the first <svg>...</svg> from a possibly noisy model response."""
    if not text:
        return None
    text = text.strip()
    # Strip code fences if present.
    text = re.sub(r"^```(?:svg|xml|html)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    m = SVG_BLOCK_RE.search(text)
    return m.group(0) if m else None


def auto_repair(text: str) -> str:
    """Best-effort fix for common LLM mistakes."""
    # Close stray unclosed self-closing tags.
    text = re.sub(r"<(path|circle|rect|line|polyline|polygon|ellipse)([^/>]*?)>", r"<\1\2/>", text)
    # Ensure single root <svg>.
    if text.count("<svg") > 1:
        first = text.find("<svg")
        last = text.rfind("</svg>")
        if first != -1 and last != -1:
            text = text[first : last + len("</svg>")]
    return text


def snap_grid(svg: str, step: float = 0.5) -> str:
    """Round numbers in path data to the nearest grid step."""
    if step <= 0:
        return svg

    def _r(match: re.Match[str]) -> str:
        v = float(match.group(0))
        snapped = round(v / step) * step
        return f"{snapped:g}"

    return NUM_RE.sub(_r, svg)


def run_svgo(svg: str) -> str:
    """Run svgo if available; otherwise return input unchanged."""
    if not shutil.which("svgo"):
        return svg
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False, encoding="utf-8") as f:
            f.write(svg)
            tmp_path = f.name
        result = subprocess.run(
            ["svgo", "--multipass", "--quiet", tmp_path, "-o", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return Path(tmp_path).read_text(encoding="utf-8")
        log.warning("svgo failed: %s", result.stderr.strip())
    except Exception as e:  # noqa: BLE001
        log.warning("svgo invocation error: %s", e)
    return svg


def postprocess(
    raw_text: str,
    *,
    target_size: int = 24,
    snap_step: float = 0.5,
    use_svgo: bool = True,
) -> PostprocessResult:
    issues: list[str] = []

    block = extract_svg_block(raw_text)
    if block is None:
        return PostprocessResult(None, False, ["no_svg_block_found"])

    block = auto_repair(block)

    try:
        parser = etree.XMLParser(recover=True, remove_comments=True)
        root = etree.fromstring(block.encode("utf-8"), parser=parser)
    except Exception as e:  # noqa: BLE001
        return PostprocessResult(None, False, [f"parse_error: {e}"])

    if root is None or not isinstance(root.tag, str) or not root.tag.endswith("svg"):
        return PostprocessResult(None, False, ["root_not_svg"])

    if not root.get("viewBox"):
        root.set("viewBox", f"0 0 {target_size} {target_size}")
        issues.append("added_default_viewbox")

    if not root.get("xmlns"):
        root.set("xmlns", SVG_NS)
        issues.append("added_xmlns")

    root.set("width", str(target_size))
    root.set("height", str(target_size))

    out = etree.tostring(root, encoding="unicode")
    out = snap_grid(out, snap_step)

    if use_svgo:
        out = run_svgo(out)

    return PostprocessResult(out, True, issues)
