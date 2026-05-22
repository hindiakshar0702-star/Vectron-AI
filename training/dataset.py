"""
Dataset loading helpers for Vectron AI training.

The training data is JSONL produced by data/build_dataset.py. Each line has:
    {"messages": [...], "meta": {...}}

We load it via HuggingFace `datasets` and apply the model's chat template at
collation time so the same JSONL works with any chat model (Qwen, Llama, Mistral).
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

log = logging.getLogger(__name__)


def load_chat_dataset(
    train_file: str | Path,
    eval_file: str | Path | None = None,
) -> DatasetDict:
    """Load JSONL into a DatasetDict with `train` and optional `eval` splits."""
    data_files: dict[str, str] = {"train": str(train_file)}
    if eval_file and Path(eval_file).exists():
        data_files["eval"] = str(eval_file)

    ds = load_dataset("json", data_files=data_files)
    log.info("Loaded dataset: %s", {k: len(v) for k, v in ds.items()})
    return ds


def attach_text_column(
    ds: Dataset,
    tokenizer,
    text_column: str = "text",
) -> Dataset:
    """Render the chat template into a flat `text` field for SFTTrainer."""

    def _render(batch):
        out: list[str] = []
        for messages in batch["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            out.append(text)
        return {text_column: out}

    return ds.map(_render, batched=True, desc="apply_chat_template", num_proc=4)
