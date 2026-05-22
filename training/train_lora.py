"""
QLoRA fine-tuning for Vectron AI - SVG generation.

Trains a LoRA adapter on top of a 4-bit quantised code-LLM (Qwen2.5-Coder by
default), using SFT on chat-formatted data produced by data/build_dataset.py.

Usage:
    python training/train_lora.py --config training/config.yaml
    python training/train_lora.py --config training/config.yaml --preset colab_t4
    python training/train_lora.py --config training/config.yaml --resume checkpoints/vectron-v0.1
"""

from __future__ import annotations

import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_lora")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge - override wins."""
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path, preset: str | None) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    presets = raw.pop("presets", {}) or {}
    cfg = raw
    if preset:
        if preset not in presets:
            raise ValueError(f"Preset '{preset}' not found. Available: {list(presets)}")
        log.info("Applying preset: %s", preset)
        cfg = deep_merge(cfg, presets[preset])
    return cfg


def build_bnb_config(qcfg: dict[str, Any]):
    from transformers import BitsAndBytesConfig

    compute_dtype = getattr(torch, qcfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=qcfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
    )


def build_lora_config(lcfg: dict[str, Any]):
    from peft import LoraConfig

    return LoraConfig(
        r=lcfg.get("r", 16),
        lora_alpha=lcfg.get("alpha", 32),
        lora_dropout=lcfg.get("dropout", 0.05),
        target_modules=lcfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lcfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap dataset (debug)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.preset)

    # --- Lazy heavy imports so --help is fast ---
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    from training.dataset import attach_text_column, load_chat_dataset

    tcfg = cfg["training"]
    set_seed(tcfg.get("seed", 42))

    # --- Tokenizer ---
    log.info("Loading tokenizer for %s", cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Data ---
    ds = load_chat_dataset(cfg["train_file"], cfg.get("eval_file"))
    if args.max_train_samples:
        ds["train"] = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))
        log.info("Capped train set to %d samples", len(ds["train"]))
    ds = ds.map(
        lambda batch: {
            "text": [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in batch["messages"]
            ]
        },
        batched=True,
        desc="apply_chat_template",
    )

    # --- Quantization + model ---
    quantization_config = (
        build_bnb_config(cfg["quantization"]) if torch.cuda.is_available() else None
    )
    log.info("Loading base model %s ...", cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if tcfg.get("bf16", True) else torch.float16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable() if tcfg.get("gradient_checkpointing", True) else None

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = build_lora_config(cfg["lora"])

    # --- TRL SFTConfig ---
    sft_args = SFTConfig(
        output_dir=cfg["output_dir"],
        run_name=cfg.get("run_name", "vectron"),
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        optim=tcfg["optim"],
        max_grad_norm=tcfg["max_grad_norm"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        eval_steps=tcfg["eval_steps"] if "eval" in ds else None,
        eval_strategy="steps" if "eval" in ds else "no",
        save_total_limit=tcfg["save_total_limit"],
        bf16=tcfg.get("bf16", True),
        fp16=tcfg.get("fp16", False),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        group_by_length=tcfg.get("group_by_length", True),
        max_length=cfg.get("max_seq_length", 2048),
        report_to=tcfg.get("report_to", "none"),
        seed=tcfg.get("seed", 42),
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    log.info("Trainable parameter summary:")
    trainer.model.print_trainable_parameters()

    log.info("Starting training - %d examples, %d epochs",
             len(ds["train"]), tcfg["num_train_epochs"])
    trainer.train(resume_from_checkpoint=args.resume)

    final_dir = Path(cfg["output_dir"]) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    log.info("Saved final adapter to %s", final_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
