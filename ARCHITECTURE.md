# Vectron AI - Architecture

## 1. High-Level System

```
                      User Prompt
                          |
                          v
          +---------------------------------+
          |  Prompt Enhancer (optional)     |
          |  "wallet icon" -> rich prompt   |
          +---------------------------------+
                          |
                          v
          +---------------------------------+
          |  Code-LLM (Qwen2.5-Coder + LoRA)|
          |  Autoregressive SVG generation  |
          +---------------------------------+
                          |
                          v
          +---------------------------------+
          |  Post-processor                  |
          |  - lxml validate                 |
          |  - svgo optimize                 |
          |  - grid snap (24/32/48 px)       |
          |  - color palette enforce         |
          +---------------------------------+
                          |
                          v
          +---------------------------------+
          |  Renderer (cairosvg)             |
          |  SVG -> PNG previews 1x/2x/3x    |
          +---------------------------------+
                          |
                          v
                Output: { svg, png, meta }
```

## 2. Why Code-LLM Beats Diffusion for Icons

SVG is essentially a domain-specific language with a small grammar:

```
<svg viewBox="0 0 24 24">
  <path d="M3 12 L12 3 L21 12 Z" fill="#1e88e5"/>
</svg>
```

Code-LLMs (Qwen2.5-Coder, DeepSeek-Coder, StarCoder2) are pretrained on billions of tokens of code, including SVG/XML in HTML files. They already understand:

- XML structure
- Path commands (M, L, C, Q, A, Z)
- Attribute conventions (fill, stroke, stroke-width)
- Common viewBox sizes

A small LoRA adapter (~30 MB) on top of a 1.5B base model is enough to specialize for icon generation.

## 3. Tokenization Strategy

We use the base model's native BPE tokenizer (no custom tokens). SVG is fed as raw text. Why?

- Custom SVG tokens require training tokenizer + base model from scratch (expensive).
- Code-LLM tokenizers already efficiently compress numeric coordinates and command letters.
- Empirically, IconShop-style custom tokens give marginal gains for 5-10x more compute.

**Training format (chat template):**

```
<|im_start|>user
Generate an SVG icon: minimal blue fintech wallet, 24x24 viewBox, flat style
<|im_end|>
<|im_start|>assistant
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <path d="..." fill="#1e88e5"/>
</svg>
<|im_end|>
```

## 4. Data Pipeline

```
  Iconify CDN  ----+
  FIGR-8       ----+----> raw SVGs (data/raw/*.svg)
  Material     ----+              |
  Lucide       ----+              v
                          svg_cleaner.py
                                  |
                                  v
                          clean SVGs (data/clean/*.svg)
                                  |
                                  v
                          caption_generator.py
                          (BLIP-2 / heuristic)
                                  |
                                  v
                          captions.jsonl
                          {path, caption, tags}
                                  |
                                  v
                          build_dataset.py
                                  |
                                  v
                          train.jsonl (chat format)
```

### SVG Cleaning Steps

1. Parse with `lxml`
2. Strip metadata, comments, scripts
3. Normalize viewBox to `0 0 24 24` (or 32, 64)
4. Round coordinates to 2 decimals
5. Remove unused defs/groups
6. Validate path syntax
7. Reject if: > 5KB, > 20 paths, contains raster `<image>`

## 5. Training

**Base model choices:**

| Model | Params | VRAM (QLoRA) | Quality |
|-------|--------|---------------|---------|
| Qwen2.5-Coder-0.5B | 500M | 4 GB | Good for PoC |
| Qwen2.5-Coder-1.5B | 1.5B | 8 GB | Recommended |
| Qwen2.5-Coder-7B | 7B | 24 GB | Production |
| DeepSeek-Coder-1.3B | 1.3B | 8 GB | Alternative |

**LoRA config:**

- rank `r = 16`
- alpha = 32
- target_modules = `["q_proj", "k_proj", "v_proj", "o_proj"]`
- 4-bit NF4 quantization (bitsandbytes)

**Training hyperparams:**

- learning_rate: 2e-4
- batch_size: 4 (per device)
- gradient_accumulation_steps: 4 -> effective batch 16
- max_seq_len: 2048
- epochs: 3
- warmup_ratio: 0.03
- lr_scheduler: cosine

## 6. Post-Processing

The model can produce malformed SVG. We have a strict validator:

1. **Syntactic**: `lxml.etree.fromstring` must succeed.
2. **Semantic**: viewBox present, at least one drawable element.
3. **Optimization**: run through `svgo` (Node) or `scour` (Python).
4. **Grid snap**: round path coordinates to nearest 0.5 px.
5. **Repair pass**: if validation fails, re-prompt the model with the broken SVG and ask to fix.

## 7. Evaluation Metrics

| Metric | What it measures | Tool |
|--------|------------------|------|
| Validity rate | % of outputs that parse as valid SVG | lxml |
| CLIP score | Prompt-image similarity (rendered) | open_clip |
| Path complexity | Avg # paths, anchors per icon | custom |
| Aesthetic score | Learned quality predictor | LAION aesthetic v2 |
| Human pref | A/B win rate vs baseline | manual / Label Studio |

## 8. Future Extensions

- **Style conditioning**: prepend `<style:flat>`, `<style:3d>` tokens
- **Color palette conditioning**: `<palette:#1e88e5,#ffffff,#0d47a1>`
- **Sketch-to-SVG**: vision encoder (SigLIP) cross-attention
- **DPO**: Direct Preference Optimization on designer-rated pairs
- **Tool use**: model can call `svgo`, `vtracer` mid-generation

## 9. Serving

- **Dev**: FastAPI + single-GPU inference
- **Prod**: vLLM or TGI for batched generation, Redis queue for async jobs
- **Edge**: GGUF-quantized model running on CPU for unlimited free tier
