"""FLUX.2 [klein] 9B text-to-image graph, in ComfyUI's ``/prompt`` (API) format.

Flattened from ComfyUI's official ``image_flux2_text_to_image_9b`` template. The
template routes width/height through utility nodes to snap them; that arithmetic
is done here in Python so the emitted graph holds only model nodes.

Graph shape (all nodes are ComfyUI core)::

    UNETLoader ------------------------------.
    CLIPLoader -> CLIPTextEncode (positive) --+-> CFGGuider
               `-> CLIPTextEncode (negative) -'        |
    RandomNoise / KSamplerSelect / Flux2Scheduler / EmptyFlux2LatentImage
                                                       |
                                    SamplerCustomAdvanced -> VAEDecode -> SaveImage

Simpler than the Ideogram 4 graph in two ways that matter to callers: one
transformer instead of a conditional/unconditional pair, and a real negative
prompt rather than zeroed-out conditioning.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

# --- Weight files -----------------------------------------------------------
# The two transformers are gated on Hugging Face; see app.py for the token.
TEXT_ENCODER = "qwen_3_8b_fp8mixed.safetensors"
# Not flux2-vae: the klein templates ship the small-decoder autoencoder.
VAE = "full_encoder_small_decoder.safetensors"

# The 9B text encoder pairs with the 9B transformer. Mixing in the 4B encoder
# degrades quality rather than failing loudly, so the pairing is fixed here.
CLIP_TYPE = "flux2"


@dataclass(frozen=True)
class Variant:
    """A checkpoint and the sampler settings it was tuned for."""

    checkpoint: str
    steps: int
    cfg: float
    description: str


# Values taken from the official templates rather than inferred: the base
# workflow ships 20 steps at cfg 5, the distilled one 4 steps at cfg 1.
# Guidance-distilled models collapse if you drive them above cfg 1.
VARIANTS: dict[str, Variant] = {
    "base": Variant(
        checkpoint="flux-2-klein-base-9b-fp8.safetensors",
        steps=20,
        cfg=5.0,
        description="Undistilled. Responds to CFG and negative prompts.",
    ),
    "distilled": Variant(
        checkpoint="flux-2-klein-9b-fp8.safetensors",
        steps=4,
        cfg=1.0,
        description="Guidance-distilled, 4 steps. Ignores CFG and negative prompts.",
    ),
}
DEFAULT_VARIANT = "base"

# --- Resolution -------------------------------------------------------------
# EmptyFlux2LatentImage packs 16x16 pixel patches, so sides must be multiples
# of 16.
MIN_SIDE = 256
MAX_SIDE = 2048
SIDE_MULTIPLE = 16

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "2:3": (2, 3),
    "3:2": (3, 2),
    "3:4": (3, 4),
    "4:3": (4, 3),
    "9:16": (9, 16),
    "16:9": (16, 9),
    "21:9": (21, 9),
}

MAX_SEED = 0xFFFFFFFFFFFFFFFF


class WorkflowError(ValueError):
    """Raised when generation parameters cannot produce a valid graph."""


def snap_side(value: int) -> int:
    """Round a side up to a multiple of 16, clamped to the trained range."""
    snapped = max(((int(value) + SIDE_MULTIPLE - 1) // SIDE_MULTIPLE) * SIDE_MULTIPLE, MIN_SIDE)
    return min(snapped, MAX_SIDE)


def resolution_for(aspect_ratio: str, megapixels: float = 1.0) -> tuple[int, int]:
    """Width/height for an aspect ratio at a pixel budget, snapped for the model."""
    if aspect_ratio not in ASPECT_RATIOS:
        raise WorkflowError(
            f"unknown aspect_ratio {aspect_ratio!r}; expected one of {sorted(ASPECT_RATIOS)}"
        )
    if megapixels <= 0:
        raise WorkflowError("megapixels must be positive")
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    scale = math.sqrt(megapixels * 1024 * 1024 / (w_ratio * h_ratio))
    return snap_side(round(w_ratio * scale)), snap_side(round(h_ratio * scale))


def random_seed() -> int:
    return random.randrange(MAX_SEED + 1)


@dataclass(frozen=True)
class GenerationParams:
    """Fully resolved sampler settings — no defaults left to the graph."""

    prompt: str
    negative_prompt: str
    variant: str
    checkpoint: str
    width: int
    height: int
    seed: int
    batch_size: int
    steps: int
    cfg: float
    sampler_name: str
    filename_prefix: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_params(
    prompt: str,
    *,
    negative_prompt: str = "",
    variant: str = DEFAULT_VARIANT,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    batch_size: int = 1,
    steps: int | None = None,
    cfg: float | None = None,
    sampler_name: str = "euler",
    filename_prefix: str = "flux2-klein",
) -> GenerationParams:
    """Apply the variant's defaults and snap dimensions; explicit values win."""
    if not prompt or not prompt.strip():
        raise WorkflowError("prompt must not be empty")
    if variant not in VARIANTS:
        raise WorkflowError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    if batch_size < 1:
        raise WorkflowError("batch_size must be at least 1")

    spec = VARIANTS[variant]
    resolved_steps = int(steps if steps is not None else spec.steps)
    if resolved_steps < 1:
        raise WorkflowError("steps must be at least 1")

    return GenerationParams(
        prompt=prompt,
        negative_prompt=negative_prompt,
        variant=variant,
        checkpoint=spec.checkpoint,
        width=snap_side(width),
        height=snap_side(height),
        seed=random_seed() if seed is None else int(seed) % (MAX_SEED + 1),
        batch_size=int(batch_size),
        steps=resolved_steps,
        cfg=float(cfg if cfg is not None else spec.cfg),
        sampler_name=sampler_name,
        filename_prefix=filename_prefix,
    )


OUTPUT_NODE_ID = "save_image"


def build_workflow(params: GenerationParams) -> dict[str, Any]:
    """Emit the API-format graph ComfyUI's ``POST /prompt`` accepts."""
    return {
        "load_unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": params.checkpoint, "weight_dtype": "default"},
            "_meta": {"title": f"FLUX.2 klein 9B ({params.variant})"},
        },
        "load_clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER,
                "type": CLIP_TYPE,
                "device": "default",
            },
            "_meta": {"title": "Qwen3 8B text encoder"},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["load_clip", 0], "text": params.prompt},
            "_meta": {"title": "Prompt"},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["load_clip", 0], "text": params.negative_prompt},
            "_meta": {"title": "Negative prompt"},
        },
        "guider": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["load_unet", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "cfg": params.cfg,
            },
            "_meta": {"title": "CFG"},
        },
        "sampler": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": params.sampler_name},
            "_meta": {"title": "Sampler"},
        },
        "sigmas": {
            "class_type": "Flux2Scheduler",
            "inputs": {
                "steps": params.steps,
                "width": params.width,
                "height": params.height,
            },
            "_meta": {"title": "FLUX.2 schedule"},
        },
        "noise": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": params.seed},
            "_meta": {"title": "Noise"},
        },
        "latent": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": params.width,
                "height": params.height,
                "batch_size": params.batch_size,
            },
            "_meta": {"title": "Empty latent"},
        },
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["sampler", 0],
                "sigmas": ["sigmas", 0],
                "latent_image": ["latent", 0],
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
