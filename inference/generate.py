"""
Generate SVG icons from a text prompt using a fine-tuned Vectron model.

Loads the base model + LoRA adapter, prompts via the chat template, samples
N variants, runs the post-processor, and (optionally) writes PNG previews.

Usage:
    python inference/generate.py \
        --prompt "minimal blue fintech wallet icon, flat style" \
        --adapter checkpoints/vectron-v0.1/final \
        --output out.svg

    # Multiple variants + preview grid:
    python inference/generate.py \
        --prompt "shield with lock, security" \
        --adapter checkpoints/vectron-v0.1/final \
        --num-variants 4 \
        --output-dir outputs/run-001
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch


log = logging.getLogger("generate")


SYSTEM_PROMPT = (
    "You are Vectron AI, an expert SVG icon generator. "
    "Always respond with a single, valid, self-contained SVG document using a 24x24 viewBox. "
    "Use clean paths, no inline scripts, no <image>, and prefer a single fill colour."
)


@dataclass
class GenerationConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    adapter: str | None = None
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    num_variants: int = 1
    seed: int | None = None
    device: str = "auto"


class VectronGenerator:
    """Lazy-loaded model wrapper. Reusable across many prompts (e.g. in API server)."""

    def __init__(self, cfg: GenerationConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading tokenizer: %s", self.cfg.base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device_map = self.cfg.device if self.cfg.device != "auto" else "auto"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        log.info("Loading base model: %s", self.cfg.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        if self.cfg.adapter:
            from peft import PeftModel

            log.info("Loading LoRA adapter from %s", self.cfg.adapter)
            model = PeftModel.from_pretrained(model, self.cfg.adapter)
            model = model.merge_and_unload()

        model.eval()
        self._model = model

    @torch.inference_mode()
    def generate(self, prompt: str, num_variants: int | None = None) -> list[str]:
        if self._model is None:
            self.load()

        n = num_variants or self.cfg.num_variants
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate an SVG icon: {prompt}"},
        ]
        chat_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(chat_text, return_tensors="pt").to(self._model.device)

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            repetition_penalty=self.cfg.repetition_penalty,
            num_return_sequences=n,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        prompt_len = inputs.input_ids.shape[1]
        completions = []
        for seq in outputs:
            text = self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            completions.append(text)
        return completions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--output", type=Path, default=None, help="Single SVG output path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for variants")
    parser.add_argument("--num-variants", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render-png", action="store_true", help="Also write PNG previews")
    args = parser.parse_args()

    if not args.output and not args.output_dir:
        args.output_dir = Path("outputs")

    cfg = GenerationConfig(
        base_model=args.base_model,
        adapter=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_variants=args.num_variants,
        seed=args.seed,
    )
    gen = VectronGenerator(cfg)
    completions = gen.generate(args.prompt, num_variants=args.num_variants)

    from inference.postprocess import postprocess

    results = []
    for i, raw in enumerate(completions):
        result = postprocess(raw)
        results.append(result)
        log.info(
            "Variant %d - valid=%s issues=%s len=%d",
            i, result.valid, result.issues, len(result.svg or "")
        )

    if args.output and len(results) >= 1 and results[0].valid:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(results[0].svg, encoding="utf-8")
        log.info("Wrote %s", args.output)
        if args.render_png:
            from inference.render import save_previews

            save_previews(results[0].svg, args.output)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(results):
            if not r.valid:
                continue
            out = args.output_dir / f"variant_{i:02d}.svg"
            out.write_text(r.svg, encoding="utf-8")
        if args.render_png and results:
            from inference.render import make_preview_grid

            grid = make_preview_grid([r.svg for r in results if r.valid])
            grid.save(args.output_dir / "preview_grid.png")
        log.info("Wrote %d variants to %s", sum(1 for r in results if r.valid), args.output_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
