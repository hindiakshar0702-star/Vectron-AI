"""
Vectron AI - FastAPI inference server.

Endpoints:
- GET  /health        -> liveness probe
- GET  /info          -> model + adapter info
- POST /generate      -> { prompt, num_variants?, temperature?, render_png? } -> JSON
- POST /generate.svg  -> same body, returns first variant as image/svg+xml

Run:
    export VECTRON_BASE_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
    export VECTRON_ADAPTER=checkpoints/vectron-v0.1/final
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from inference.generate import GenerationConfig, VectronGenerator
from inference.postprocess import postprocess


log = logging.getLogger("api.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---- request / response schemas ---------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, examples=["minimal blue wallet icon"])
    num_variants: int = Field(1, ge=1, le=8)
    temperature: float = Field(0.7, ge=0.0, le=1.5)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_new_tokens: int = Field(1024, ge=64, le=2048)
    seed: Optional[int] = None
    render_png: bool = False
    snap_grid: float = Field(0.5, ge=0.0, le=2.0)


class Variant(BaseModel):
    svg: Optional[str]
    valid: bool
    issues: list[str]
    png_base64: Optional[str] = None


class GenerateResponse(BaseModel):
    prompt: str
    variants: list[Variant]
    elapsed_ms: float
    model: str
    adapter: Optional[str] = None


# ---- application lifecycle ---------------------------------------------------


_generator: VectronGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _generator
    cfg = GenerationConfig(
        base_model=os.getenv("VECTRON_BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        adapter=os.getenv("VECTRON_ADAPTER") or None,
    )
    log.info("Initialising Vectron generator: %s + adapter=%s", cfg.base_model, cfg.adapter)
    _generator = VectronGenerator(cfg)
    if os.getenv("VECTRON_EAGER_LOAD", "1") == "1":
        _generator.load()
        log.info("Model preloaded.")
    yield
    _generator = None


app = FastAPI(
    title="Vectron AI",
    description="AI-powered SVG icon generation API.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---- helpers -----------------------------------------------------------------


def _gen() -> VectronGenerator:
    if _generator is None:
        raise HTTPException(status_code=503, detail="generator not initialised")
    return _generator


def _render_png_b64(svg: str, size: int = 256) -> str | None:
    try:
        from inference.render import svg_to_png_bytes

        return base64.b64encode(svg_to_png_bytes(svg, width=size, height=size)).decode("ascii")
    except Exception as e:  # noqa: BLE001
        log.warning("png render failed: %s", e)
        return None


# ---- routes ------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, str | None]:
    g = _gen()
    return {"base_model": g.cfg.base_model, "adapter": g.cfg.adapter}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    g = _gen()
    g.cfg.temperature = req.temperature
    g.cfg.top_p = req.top_p
    g.cfg.max_new_tokens = req.max_new_tokens
    g.cfg.seed = req.seed

    t0 = time.perf_counter()
    raw = g.generate(req.prompt, num_variants=req.num_variants)
    variants: list[Variant] = []
    for r in raw:
        result = postprocess(r, snap_step=req.snap_grid)
        png_b64 = (
            _render_png_b64(result.svg)
            if req.render_png and result.valid and result.svg
            else None
        )
        variants.append(
            Variant(
                svg=result.svg,
                valid=result.valid,
                issues=result.issues,
                png_base64=png_b64,
            )
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return GenerateResponse(
        prompt=req.prompt,
        variants=variants,
        elapsed_ms=elapsed_ms,
        model=g.cfg.base_model,
        adapter=g.cfg.adapter,
    )


@app.post("/generate.svg")
def generate_svg(req: GenerateRequest) -> Response:
    """Return the first valid variant as raw image/svg+xml."""
    g = _gen()
    g.cfg.temperature = req.temperature
    g.cfg.top_p = req.top_p
    g.cfg.max_new_tokens = req.max_new_tokens
    g.cfg.seed = req.seed

    raw = g.generate(req.prompt, num_variants=max(req.num_variants, 1))
    for r in raw:
        result = postprocess(r, snap_step=req.snap_grid)
        if result.valid and result.svg:
            return Response(content=result.svg, media_type="image/svg+xml")
    raise HTTPException(status_code=422, detail="no valid SVG produced")
