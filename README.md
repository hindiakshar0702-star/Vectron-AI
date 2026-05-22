# Vectron AI

> **Geometry-aware, symbol-reasoning AI icon generator that produces production-ready SVG from text prompts.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4+-red.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## Vision

Vectron AI ek next-generation hybrid system hai jo **language reasoning + geometric intelligence + vector code generation** ko combine karta hai. Diffusion-based raster generators ki jagah, Vectron SVG ko **code** ki tarah treat karta hai aur fine-tuned code-LLM se directly clean, scalable, editable vector icons produce karta hai.

**Inspired by:** [StarVector](https://github.com/joanrod/star-vector) (ServiceNow), [IconShop](https://arxiv.org/abs/2304.14400), [SVGen](https://arxiv.org/html/2508.09168v1).

---

## Why SVG-as-Code?

| Approach | Pros | Cons |
|----------|------|------|
| **Diffusion + vectorize** (SDXL → VTracer) | Fast PoC | Messy paths, no editability |
| **Vector-space diffusion** (SVGFusion) | Native vector | Very expensive to train |
| **Code-LLM fine-tune** ⭐ Vectron's choice | Clean SVG, editable, small model fits in 24GB | Needs good tokenization |

---

## Project Structure

```
Vectron-AI/
├── README.md              # You are here
├── ARCHITECTURE.md        # Deep technical doc
├── requirements.txt       # Pinned Python deps
├── scripts/
│   └── check_environment.py
├── data/
│   ├── download_iconify.py     # Pull free icon datasets
│   ├── svg_cleaner.py          # Normalize + simplify SVG
│   ├── caption_generator.py    # Auto-caption with BLIP-2
│   └── build_dataset.py        # End-to-end data pipeline
├── training/
│   ├── train_lora.py           # QLoRA fine-tuning entry point
│   ├── dataset.py              # PyTorch Dataset class
│   └── config.yaml             # Hyperparameters
├── inference/
│   ├── generate.py             # Prompt -> SVG
│   ├── postprocess.py          # Validate + optimize
│   └── render.py               # SVG -> PNG preview
├── api/
│   └── server.py               # FastAPI inference server
└── notebooks/
    └── 02_train_colab.ipynb    # One-click Colab training
```

---

## Quickstart

### 1. Setup

> **For real training, use Colab or a Linux machine with a GPU.**
> The local install is fine for code editing and data preparation only.
> See the [Hardware Requirements](#hardware-requirements) section.

#### Linux / macOS

```bash
git clone https://github.com/hindiakshar0702-star/Vectron-AI.git
cd Vectron-AI
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_environment.py
```

#### Windows (PowerShell)

PowerShell does not support `&&` and uses `Scripts\Activate.ps1` instead of `bin/activate`.

```powershell
git clone https://github.com/hindiakshar0702-star/Vectron-AI.git
cd Vectron-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_environment.py
```

If `Activate.ps1` is blocked by policy, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

If `python` opens the Microsoft Store, install real Python from
[python.org](https://www.python.org/downloads/) and tick **"Add Python to PATH"**
during install.

> **Windows + training caveats:** `bitsandbytes` (used for 4-bit quantization)
> is not officially supported on native Windows, and `cairosvg` needs GTK runtime
> libraries. For training on Windows, use **WSL2** (Ubuntu) or run training in Colab.

### 2. Get data (~200K free icons)

```bash
python data/download_iconify.py --output data/raw --max-icons 50000
python data/svg_cleaner.py --input data/raw --output data/clean
python data/caption_generator.py --input data/clean --output data/captions.jsonl
python data/build_dataset.py --captions data/captions.jsonl --output data/train.jsonl
```

### 3. Train (Colab T4 friendly)

```bash
python training/train_lora.py --config training/config.yaml
```

For Colab, open `notebooks/02_train_colab.ipynb` and click **Run All**. Free T4 GPU is enough for `Qwen2.5-Coder-0.5B`.

### 4. Generate

```bash
python inference/generate.py --prompt "minimal blue fintech wallet icon" --output out.svg
```

### 5. Serve as API

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
# POST http://localhost:8000/generate  {"prompt": "..."}
```

---

## Hardware Requirements

| Setup | GPU | Model size | Training time |
|-------|-----|-----------|---------------|
| Colab Free (T4) | 16 GB | 0.5B (QLoRA) | ~6 hr / 10K samples |
| Colab Pro (A100) | 40 GB | 1.5B (QLoRA) | ~3 hr / 50K samples |
| RunPod (A100 80GB) | 80 GB | 7B (QLoRA) | ~12 hr / 200K samples |

---

## Roadmap

- [x] Phase 0 — Project scaffold
- [ ] Phase 1 — Iconify dataset + Qwen2.5-Coder-0.5B QLoRA
- [ ] Phase 2 — Synthetic data via GPT-4o + SDXL pipeline
- [ ] Phase 3 — Style intelligence (multi-style classifier conditioning)
- [ ] Phase 4 — Brand DNA extractor
- [ ] Phase 5 — RLHF from designer feedback
- [ ] Phase 6 — Figma plugin + public API

---

## License

Apache 2.0 - see [LICENSE](LICENSE).

Datasets pulled by `data/download_iconify.py` retain their original licenses (mostly MIT / Apache / OFL).

---

## Acknowledgments

Built on the shoulders of: HuggingFace, Qwen, StarVector, IconShop, Iconify, Material Icons, Lucide, Tabler.
