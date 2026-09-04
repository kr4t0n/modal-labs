"""RedCraft v3 (Krea 2) text-to-image graph, in ComfyUI's ``/prompt`` format.

The fourth Krea 2 service here, and structurally the same as the other three: a
diffusion model only — no text encoder, no VAE — loaded through `UNETLoader`
beside the Qwen3-VL-4B encoder and the Qwen-Image autoencoder.

Graph shape, the Krea 2 turbo reference template::

    UNETLoader ---------------------------------------------.
    CLIPLoader -> CLIPTextEncode (positive) -----------------+-> KSampler
                            `-> ConditioningZeroOut ---------'      |
    EmptyLatentImage --------------------------------------'       |
                                                    VAEDecode -> SaveImage

What differs is only where the sampler defaults come from: this edition's notes
publish a recipe, so they are the author's rather than a template fallback.
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
# Renamed on the way into the Volume. Upstream calls every precision of this
# version `redcraftHybridH3A2A_30Krea2.safetensors` — the same name for four
# different files — so the destination in app.py is what makes the graph
# unambiguous about which one is loaded.
DIFFUSION_MODEL = "redcraft_v3_krea2_fp8.safetensors"
# Shared verbatim with the ultra, finepornv4 and redgpt2gpt services: same base
# model, same companions. A test asserts they stay in step.
TEXT_ENCODER = "qwen3vl_4b_fp8_scaled.safetensors"
CLIP_TYPE = "krea2"
VAE = "qwen_image_vae.safetensors"

# --- Sampling ---------------------------------------------------------------
# Published by the author, not inferred: the version notes read
# "ER_SDE / Euler | Simple | CFG =1 | 8-12 Steps".
#
# `euler` rather than the `er_sde` listed first, because the card offers the two
# interchangeably and euler is present in every ComfyUI build. Whether a given
# deployment has er_sde is answerable only against a live one — see
# `client.py validate` — and a default that fails to resolve would break every
# request that omits a sampler. The node offers er_sde in its dropdown.
DEFAULT_SAMPLER = "euler"
ALTERNATE_SAMPLER = "er_sde"
DEFAULT_SCHEDULER = "simple"
DEFAULT_CFG = 1.0
# Midpoint of the published 8-12 band.
DEFAULT_STEPS = 10
STEPS_RANGE = (8, 12)

# Reported through /defaults so a caller can tell a published recipe from a
# template fallback without reading the source. redgpt2gpt, the other service
# over this same base, has no published settings and says so there.
DEFAULTS_SOURCE = "the RedCraft v3 (Krea2) version notes, not a ComfyUI template"


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
    filename_prefix: str = "redcraftv3",
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
            "_meta": {"title": "RedCraft v3 (Krea 2)"},
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

    Same reasoning as the other Krea 2 services: the card specifies cfg 1, where
    the negative branch is never consulted, so encoding real text there is only
    meaningful alongside a raised cfg and stays opt-in.
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
