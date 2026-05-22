"""
Clean and normalize SVG icons for training.

Steps:
1. Parse with lxml; reject malformed.
2. Strip metadata, comments, scripts, foreignObject, raster <image>.
3. Normalize viewBox to 0 0 24 24 (configurable).
4. Round numeric path coordinates to N decimals.
5. Remove unused defs / empty groups.
6. Reject if too large (> max_bytes) or too complex (> max_paths).
7. Write to output dir preserving prefix/name structure.

Usage:
    python data/svg_cleaner.py --input data/raw --output data/clean --target-size 24
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import jsonlines
from lxml import etree
from tqdm import tqdm


SVG_NS = "http://www.w3.org/2000/svg"
NSMAP = {"svg": SVG_NS}

# Elements we drop wholesale.
DROP_TAGS = {
    f"{{{SVG_NS}}}metadata",
    f"{{{SVG_NS}}}script",
    f"{{{SVG_NS}}}foreignObject",
    f"{{{SVG_NS}}}image",
    f"{{{SVG_NS}}}a",
    f"{{{SVG_NS}}}title",
    f"{{{SVG_NS}}}desc",
}

# Attributes safe to drop (presentation only / editor metadata).
DROP_ATTRS = {
    "id",
    "class",
    "data-name",
    "{http://www.inkscape.org/namespaces/inkscape}label",
    "{http://www.inkscape.org/namespaces/inkscape}groupmode",
    "{http://www.inkscape.org/namespaces/inkscape}version",
    "{http://www.w3.org/XML/1998/namespace}space",
    "{http://creativecommons.org/ns#}license",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("svg_cleaner")


_NUM_RE = re.compile(r"-?\d+\.\d+(?:e-?\d+)?")


def round_numbers(text: str, decimals: int) -> str:
    def _r(match: re.Match[str]) -> str:
        return f"{float(match.group(0)):.{decimals}f}".rstrip("0").rstrip(".")

    return _NUM_RE.sub(_r, text)


def strip_namespace_recursive(elem: etree._Element) -> None:
    """Remove non-svg namespaces (inkscape, sodipodi, etc) recursively."""
    for el in elem.iter():
        if not isinstance(el.tag, str):
            continue
        # Drop attributes from non-svg namespaces.
        for attr in list(el.attrib):
            if attr.startswith("{") and not attr.startswith(f"{{{SVG_NS}}}"):
                if attr in DROP_ATTRS or "inkscape" in attr or "sodipodi" in attr:
                    del el.attrib[attr]
            elif attr in DROP_ATTRS:
                del el.attrib[attr]


def clean_svg(
    svg_text: str,
    target_size: int = 24,
    decimals: int = 2,
    max_bytes: int = 5000,
    max_paths: int = 30,
) -> tuple[str | None, str | None]:
    """Returns (cleaned_svg, reason_if_rejected)."""
    try:
        parser = etree.XMLParser(remove_comments=True, remove_pis=True, recover=False, huge_tree=False)
        root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)
    except Exception as e:  # noqa: BLE001
        return None, f"parse_error: {e}"

    if not isinstance(root.tag, str) or not root.tag.endswith("svg"):
        return None, "not_an_svg"

    # Drop forbidden tags.
    for tag in DROP_TAGS:
        for el in root.findall(f".//{tag}"):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

    strip_namespace_recursive(root)

    # Normalize viewBox.
    view_box = root.get("viewBox")
    if not view_box:
        w = root.get("width", str(target_size))
        h = root.get("height", str(target_size))
        try:
            view_box = f"0 0 {float(w):g} {float(h):g}"
        except ValueError:
            view_box = f"0 0 {target_size} {target_size}"
    root.set("viewBox", view_box)

    # Force standard dimensions.
    root.set("width", str(target_size))
    root.set("height", str(target_size))
    root.set("xmlns", SVG_NS)

    # Count drawable elements.
    drawables = root.findall(".//svg:path", NSMAP) + \
        root.findall(".//svg:circle", NSMAP) + \
        root.findall(".//svg:rect", NSMAP) + \
        root.findall(".//svg:polygon", NSMAP) + \
        root.findall(".//svg:polyline", NSMAP) + \
        root.findall(".//svg:line", NSMAP) + \
        root.findall(".//svg:ellipse", NSMAP)
    if len(drawables) == 0:
        return None, "no_drawable_elements"
    if len(drawables) > max_paths:
        return None, f"too_complex: {len(drawables)} elements"

    # Serialize.
    out = etree.tostring(root, encoding="unicode")
    out = round_numbers(out, decimals)
    # Compact whitespace.
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"> <", "><", out)

    if len(out.encode("utf-8")) > max_bytes:
        return None, f"too_large: {len(out)} bytes"

    return out, None


def process_directory(
    input_dir: Path,
    output_dir: Path,
    target_size: int,
    decimals: int,
    max_bytes: int,
    max_paths: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir.parent / "clean_report.jsonl"

    svgs = list(input_dir.rglob("*.svg"))
    log.info("Found %d SVG files in %s", len(svgs), input_dir)

    accepted = 0
    rejected = 0
    reasons: dict[str, int] = {}

    with jsonlines.open(report_path, mode="w") as writer:
        for svg_path in tqdm(svgs, desc="cleaning"):
            try:
                text = svg_path.read_text(encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                writer.write({"path": str(svg_path), "status": "read_error", "reason": str(e)})
                rejected += 1
                continue

            cleaned, reason = clean_svg(text, target_size, decimals, max_bytes, max_paths)
            rel = svg_path.relative_to(input_dir)
            if cleaned is None:
                rejected += 1
                reasons[reason or "unknown"] = reasons.get(reason or "unknown", 0) + 1
                writer.write({"path": str(rel), "status": "rejected", "reason": reason})
                continue

            out_path = output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(cleaned, encoding="utf-8")
            accepted += 1
            writer.write({"path": str(rel), "status": "ok", "bytes": len(cleaned)})

    log.info("Cleaning done. Accepted: %d  Rejected: %d", accepted, rejected)
    if reasons:
        log.info("Top rejection reasons:")
        for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
            log.info("  %6d  %s", c, r)
    log.info("Report: %s", report_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("data/clean"))
    p.add_argument("--target-size", type=int, default=24)
    p.add_argument("--decimals", type=int, default=2)
    p.add_argument("--max-bytes", type=int, default=5000)
    p.add_argument("--max-paths", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        target_size=args.target_size,
        decimals=args.decimals,
        max_bytes=args.max_bytes,
        max_paths=args.max_paths,
    )


if __name__ == "__main__":
    main()
