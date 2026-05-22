"""
Combine cleaned SVGs + captions into a chat-formatted JSONL ready for SFT.

Output format (one JSON object per line):
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "Generate an SVG icon: ..."},
        {"role": "assistant", "content": "<svg ...>...</svg>"}
      ],
      "meta": {"path": "...", "collection": "...", "tags": [...]}
    }

This format is consumed directly by trl.SFTTrainer with `dataset_kwargs={"add_special_tokens": False}`
because we apply the chat template ourselves at training time.

Usage:
    python data/build_dataset.py \
        --svg-dir data/clean \
        --captions data/captions.jsonl \
        --output data/train.jsonl \
        --eval-output data/eval.jsonl \
        --eval-fraction 0.02
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import jsonlines
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are Vectron AI, an expert SVG icon generator. "
    "Always respond with a single, valid, self-contained SVG document using a 24x24 viewBox. "
    "Use clean paths, no inline scripts, no <image>, and prefer a single fill colour."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_dataset")


def load_captions(path: Path) -> dict[str, dict]:
    captions: dict[str, dict] = {}
    with jsonlines.open(path) as reader:
        for row in reader:
            captions[row["path"]] = row
    log.info("Loaded %d captions", len(captions))
    return captions


def make_user_prompt(caption_row: dict) -> str:
    prompt = caption_row.get("prompt") or caption_row.get("heuristic_caption") or "an icon"
    return f"Generate an SVG icon: {prompt}"


def build(
    svg_dir: Path,
    captions_path: Path,
    output_path: Path,
    eval_output: Path | None,
    eval_fraction: float,
    seed: int,
) -> None:
    captions = load_captions(captions_path)
    rng = random.Random(seed)
    rows: list[dict] = []
    skipped = 0

    for rel_path, cap in tqdm(captions.items(), desc="building"):
        svg_path = svg_dir / rel_path
        if not svg_path.exists():
            skipped += 1
            continue
        try:
            svg_text = svg_path.read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if not svg_text.startswith("<svg"):
            skipped += 1
            continue

        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_prompt(cap)},
                {"role": "assistant", "content": svg_text},
            ],
            "meta": {
                "path": rel_path,
                "collection": cap.get("collection", ""),
                "tags": cap.get("tags", []),
            },
        })

    rng.shuffle(rows)

    if eval_output and eval_fraction > 0:
        n_eval = max(50, int(len(rows) * eval_fraction))
        eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    else:
        eval_rows, train_rows = [], rows

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d training rows to %s", len(train_rows), output_path)

    if eval_rows:
        eval_output.parent.mkdir(parents=True, exist_ok=True)
        with eval_output.open("w", encoding="utf-8") as f:
            for r in eval_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("Wrote %d eval rows to %s", len(eval_rows), eval_output)

    if skipped:
        log.warning("Skipped %d rows (missing SVG / bad content)", skipped)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--svg-dir", type=Path, default=Path("data/clean"))
    p.add_argument("--captions", type=Path, default=Path("data/captions.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--eval-output", type=Path, default=Path("data/eval.jsonl"))
    p.add_argument("--eval-fraction", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build(
        svg_dir=args.svg_dir,
        captions_path=args.captions,
        output_path=args.output,
        eval_output=args.eval_output,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
