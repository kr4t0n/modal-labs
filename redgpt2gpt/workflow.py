"""RedGPT2 (Krea 2) text-to-image graph, in ComfyUI's ``/prompt`` format.

The third Krea 2 service here, and structurally the same as the other two: a
diffusion model only — no text encoder, no VAE — loaded through `UNETLoader`
beside the Qwen3-VL-4B encoder and the Qwen-Image autoencoder.

Graph shape, the Krea 2 turbo reference template::

    UNETLoader ---------------------------------------------.
    CLIPLoader -> CLIPTextEncode (positive) -----------------+-> KSampler
                            `-> ConditioningZeroOut ---------'      |
    EmptyLatentImage --------------------------------------'       |
                                                    VAEDecode -> SaveImage

**This is the single-model edition.** The upstream listing is titled
"Alternating Evaluation" and its model card describes a two-file scheme — a
high-noise and a low-noise checkpoint sampled alternately in a 4H+6L
configuration. That applies to the *other* version on that page. The edition
served here ships one file and samples conventionally, which is why this graph
has a single `UNETLoader`. See AGENTS.md before "fixing" that.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from comfyui_modal.geometry import (
    ASPECT_RATIOS,
    MAX_SEED,
    MAX_SIDE,
    MIN_SIDE,
    SIDE_MULTIPLE,
    WorkflowError,
    normalise_seed,
    resolution_for,
    snap_side,
)

__all__ = [
    "ASPECT_RATIOS",
    "MAX_SEED",
    "MAX_SIDE",
    "MIN_SIDE",
    "OUTPUT_NODE_ID",
    "SIDE_MULTIPLE",
    "GenerationParams",
    "WorkflowError",
    "build_workflow",
    "resolution_for",
    "resolve_params",
    "snap_side",
]

# --- Weight files -----------------------------------------------------------
# Renamed on the way into the Volume: upstream calls this
# `redgpt2Krea2Turbo_krea2GPT.safetensors`, which does not say which of the
# page's several editions it is. The destination in app.py is what the graph
# sees, so the ambiguity stops at the download.
DIFFUSION_MODEL = "redgpt2_krea2_gpt_fp8.safetensors"
# Shared verbatim with the ultra and finepornv4 services: same base model, same
# companions. A test asserts all three stay in step.
TEXT_ENCODER = "qwen3vl_4b_fp8_scaled.safetensors"
CLIP_TYPE = "krea2"
VAE = "qwen_image_vae.safetensors"

# --- Sampling ---------------------------------------------------------------
# Unlike finepornv4, whose card publishes a per-version recipe, this edition's
# notes cover training method and licensing but state no sampler settings. So
# these are ComfyUI's official Krea 2 turbo template values, exactly as the
# ultra service uses — the honest default when upstream is silent, and flagged
# as such in the README rather than presented as the author's recommendation.
DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"

# Where the defaults come from, reported through /defaults so a caller can tell
# a template default from a published one without reading the source.
DEFAULTS_SOURCE = "ComfyUI's official Krea 2 turbo template; upstream publishes no settings"


@dataclass(frozen=True)
class GenerationParams:
    """Fully resolved sampler settings — no defaults left to the graph."""

    prompt: str
    negative_prompt: str
    width: int
    height: int
    seed: int
    batch_size: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    filename_prefix: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_params(
    prompt: str,
    *,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    batch_size: int = 1,
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    sampler_name: str = DEFAULT_SAMPLER,
    scheduler: str = DEFAULT_SCHEDULER,
    denoise: float = 1.0,
    filename_prefix: str = "redgpt2gpt",
) -> GenerationParams:
    """Validate and snap the request into a fully specified parameter set."""
    if not prompt or not prompt.strip():
        raise WorkflowError("prompt must not be empty")
    if batch_size < 1:
        raise WorkflowError("batch_size must be at least 1")
    if steps < 1:
        raise WorkflowError("steps must be at least 1")
    if not 0.0 <= denoise <= 1.0:
        raise WorkflowError("denoise must be between 0 and 1")

    return GenerationParams(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=snap_side(width),
        height=snap_side(height),
        seed=normalise_seed(seed),
        batch_size=int(batch_size),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=sampler_name,
        scheduler=scheduler,
        denoise=float(denoise),
        filename_prefix=filename_prefix,
    )


OUTPUT_NODE_ID = "save_image"
NEGATIVE_NODE_ID = "negative"


def build_workflow(params: GenerationParams) -> dict[str, Any]:
    """Emit the API-format graph ComfyUI's ``POST /prompt`` accepts."""
    graph: dict[str, Any] = {
        "load_unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
            "_meta": {"title": "RedGPT2 GPT edition (Krea 2)"},
        },
        "load_clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER,
                "type": CLIP_TYPE,
                "device": "default",
            },
            "_meta": {"title": "Qwen3-VL 4B text encoder"},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["load_clip", 0], "text": params.prompt},
            "_meta": {"title": "Prompt"},
        },
        NEGATIVE_NODE_ID: _negative_node(params),
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": params.width,
                "height": params.height,
                "batch_size": params.batch_size,
            },
            "_meta": {"title": "Empty latent"},
        },
        "sample": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["load_unet", 0],
                "seed": params.seed,
                "steps": params.steps,
                "cfg": params.cfg,
                "sampler_name": params.sampler_name,
                "scheduler": params.scheduler,
                "positive": ["positive", 0],
                "negative": [NEGATIVE_NODE_ID, 0],
                "latent_image": ["latent", 0],
                "denoise": params.denoise,
            },
            "_meta": {"title": "Sample"},
        },
        "load_vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE},
            "_meta": {"title": "VAE"},
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sample", 0], "vae": ["load_vae", 0]},
            "_meta": {"title": "Decode"},
        },
        OUTPUT_NODE_ID: {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["decode", 0],
                "filename_prefix": params.filename_prefix,
            },
            "_meta": {"title": "Save"},
        },
    }
    return graph


def _negative_node(params: GenerationParams) -> dict[str, Any]:
    """Zeroed conditioning by default, a real encode when text is supplied.

    Same reasoning as the other two Krea 2 services: turbo samples at cfg 1,
    where the negative branch is never consulted, so encoding real text there is
    only meaningful alongside a raised cfg and stays opt-in.
    """
    if params.negative_prompt.strip():
        return {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["load_clip", 0], "text": params.negative_prompt},
            "_meta": {"title": "Negative prompt"},
        }
    return {
        "class_type": "ConditioningZeroOut",
        "inputs": {"conditioning": ["positive", 0]},
        "_meta": {"title": "Zeroed conditioning"},
    }
