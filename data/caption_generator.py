"""
Generate text captions for SVG icons.

Two modes:
- "heuristic" (default, no model needed): builds a caption from filename + collection
  + category, e.g. "wallet-outline" in "mdi" -> "minimal outline wallet icon, material design style".
  Fast (~10K icons/sec), zero GPU. Good for quickly bootstrapping training.

- "blip2": render SVG to PNG and caption with BLIP-2. Higher quality but ~1 sec/icon
  on a T4 GPU. Use for final dataset.

Output: data/captions.jsonl with one line per icon:
    {"path": "mdi/wallet.svg", "prompt": "...", "tags": [...], "collection": "mdi"}

Usage:
    python data/caption_generator.py --input data/clean --output data/captions.jsonl
    python data/caption_generator.py --input data/clean --output data/captions.jsonl --mode blip2
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Callable

import jsonlines
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("caption_generator")


# Lightweight semantic dictionary so heuristic captions feel natural.
COLLECTION_STYLES = {
    "mdi": "material design style",
    "ic": "google material icon style",
    "lucide": "minimal lucide line style",
    "heroicons": "tailwind heroicons style",
    "tabler": "tabler outline style",
    "ph": "phosphor flexible style",
    "carbon": "ibm carbon enterprise style",
    "fluent": "microsoft fluent style",
    "solar": "modern solar style",
}

VARIANT_KEYWORDS = {
    "outline": "outline",
    "filled": "filled",
    "fill": "filled",
    "solid": "solid",
    "duotone": "duotone",
    "bold": "bold",
    "thin": "thin stroke",
    "regular": "regular weight",
    "round": "rounded corners",
    "rounded": "rounded corners",
    "sharp": "sharp corners",
    "two-tone": "two-tone",
    "twotone": "two-tone",
    "circle": "in a circle",
    "square": "in a square",
    "off": "disabled",
}


def split_name(stem: str) -> list[str]:
    """Split 'wallet-outline-bold' or 'walletOutlineBold' into tokens."""
    tokens = re.split(r"[-_/\s]+", stem)
    expanded: list[str] = []
    for tok in tokens:
        # camelCase -> camel Case
        parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", tok)
        expanded.extend(p.lower() for p in parts if p)
    return [t for t in expanded if t]


def heuristic_caption(prefix: str, name: str, category: str | None = None) -> tuple[str, list[str]]:
    tokens = split_name(name)
    variant_words: list[str] = []
    subject_words: list[str] = []
    for t in tokens:
        if t in VARIANT_KEYWORDS:
            variant_words.append(VARIANT_KEYWORDS[t])
        elif t.isdigit():
            continue
        else:
            subject_words.append(t)

    subject = " ".join(subject_words) or "abstract"
    style = COLLECTION_STYLES.get(prefix, "")
    variant = ", ".join(variant_words)
    pieces = [f"{subject} icon"]
    if variant:
        pieces.append(variant)
    if category:
        pieces.append(f"category: {category}")
    if style:
        pieces.append(style)
    caption = ", ".join(pieces)

    tags = list(dict.fromkeys(subject_words + variant_words + ([category] if category else [])))
    return caption, tags


def build_blip2_captioner() -> Callable[[Path], str]:
    """Loads BLIP-2 lazily; returns fn(svg_path) -> caption."""
    import io
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    import cairosvg
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log.info("Loading BLIP-2 (Salesforce/blip2-opt-2.7b) on %s ...", device)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=dtype
    ).to(device)
    model.eval()

    @torch.inference_mode()
    def _caption(svg_path: Path) -> str:
        png_bytes = cairosvg.svg2png(
            url=str(svg_path), output_width=224, output_height=224,
            background_color="white",
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        inputs = processor(
            images=img,
            text="a vector icon of",
            return_tensors="pt",
        ).to(device, dtype=dtype if device == "cuda" else torch.float32)
        out = model.generate(**inputs, max_new_tokens=24)
        text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        return f"{text} icon" if "icon" not in text else text

    return _caption


def generate_captions(
    input_dir: Path,
    output_path: Path,
    mode: str,
    limit: int | None,
) -> None:
    svgs = list(input_dir.rglob("*.svg"))
    if limit:
        svgs = svgs[:limit]
    log.info("Captioning %d SVGs (mode=%s)", len(svgs), mode)

    blip2_fn = build_blip2_captioner() if mode == "blip2" else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode="w") as writer:
        for svg_path in tqdm(svgs, desc="captioning"):
            rel = svg_path.relative_to(input_dir)
            parts = rel.parts
            prefix = parts[0] if len(parts) > 1 else "unknown"
            name = svg_path.stem

            heur_caption, tags = heuristic_caption(prefix, name)
            row = {
                "path": str(rel),
                "collection": prefix,
                "name": name,
                "prompt": heur_caption,
                "heuristic_caption": heur_caption,
                "tags": tags,
            }

            if blip2_fn is not None:
                try:
                    blip_caption = blip2_fn(svg_path)
                    row["blip2_caption"] = blip_caption
                    # Combine: BLIP-2 subject + collection style.
                    style = COLLECTION_STYLES.get(prefix, "")
                    row["prompt"] = f"{blip_caption}, {style}".rstrip(", ") if style else blip_caption
                except Exception as e:  # noqa: BLE001
                    log.warning("BLIP-2 failed on %s: %s", svg_path, e)

            writer.write(row)

    log.info("Wrote captions to %s", output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/clean"))
    p.add_argument("--output", type=Path, default=Path("data/captions.jsonl"))
    p.add_argument("--mode", choices=["heuristic", "blip2"], default="heuristic")
    p.add_argument("--limit", type=int, default=None, help="Cap number of icons (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_captions(
        input_dir=args.input,
        output_path=args.output,
        mode=args.mode,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
