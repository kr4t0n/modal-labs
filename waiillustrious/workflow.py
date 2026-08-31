"""WAI-illustrious-SDXL text-to-image graph, in ComfyUI's ``/prompt`` format.

Unlike the other two services this is a *single-file SDXL checkpoint*: the UNet,
both CLIP encoders and the VAE live in one `.safetensors`, so it loads through
`CheckpointLoaderSimple` rather than separate UNet/CLIP/VAE loaders. The graph is
plain SDXL — `KSampler` rather than the custom-sampler chains the FLUX.2 and
Ideogram 4 graphs need.

Graph shape (all nodes are ComfyUI core)::

    CheckpointLoaderSimple -> CLIPSetLastLayer -> CLIPTextEncode (positive)
             |    |                          `-> CLIPTextEncode (negative)
             |    `------------------------------------------.
    EmptyLatentImage --------------------------------------. |
                                                            KSampler
                                                               |
                                                     VAEDecode -> SaveImage
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

# Re-exported so callers and tests can treat this module as the single place
# that describes one model's graph, geometry included.
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

# --- Weight file ------------------------------------------------------------
# One fp16 checkpoint in the original SDXL (`conditioner.embedders.*`) layout,
# fetched from Civitai; see app.py.
CHECKPOINT = "waiIllustriousSDXL_v170.safetensors"

# --- Sampling ---------------------------------------------------------------
# The author publishes no recommended settings, so these are the community
# conventions for booru-tagged Illustrious finetunes. Every one is overridable
# per request; none is load-bearing for correctness.
DEFAULT_STEPS = 28
DEFAULT_CFG = 5.0
DEFAULT_SAMPLER = "euler_ancestral"
DEFAULT_SCHEDULER = "normal"

# Anime SDXL finetunes are trained against the penultimate CLIP layer. Leaving
# this at -1 does not error, it just produces noticeably worse prompt adherence.
DEFAULT_CLIP_SKIP = -2

# The standard Danbooru negative. Supply "" to opt out entirely.
DEFAULT_NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "normal quality, jpeg artifacts, signature, watermark, username, blurry"
)

# SDXL is trained around 1 megapixel. The shared geometry allows up to 2048 a
# side; going much beyond ~1 MP total tends to duplicate subjects.
NATIVE_MEGAPIXELS = 1.0


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
    clip_skip: int
    denoise: float
    filename_prefix: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_params(
    prompt: str,
    *,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    batch_size: int = 1,
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    sampler_name: str = DEFAULT_SAMPLER,
    scheduler: str = DEFAULT_SCHEDULER,
    clip_skip: int = DEFAULT_CLIP_SKIP,
    denoise: float = 1.0,
    filename_prefix: str = "wai-illustrious",
) -> GenerationParams:
    """Validate and snap the request into a fully specified parameter set."""
    if not prompt or not prompt.strip():
        raise WorkflowError("prompt must not be empty")
    if batch_size < 1:
        raise WorkflowError("batch_size must be at least 1")
    if steps < 1:
        raise WorkflowError("steps must be at least 1")
    # ComfyUI's CLIPSetLastLayer accepts -24..-1; anything else fails at queue
    # time with a less obvious message than this one.
    if not -24 <= clip_skip <= -1:
        raise WorkflowError(f"clip_skip must be between -24 and -1, got {clip_skip}")
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
        clip_skip=int(clip_skip),
        denoise=float(denoise),
        filename_prefix=filename_prefix,
    )


OUTPUT_NODE_ID = "save_image"


def build_workflow(params: GenerationParams) -> dict[str, Any]:
    """Emit the API-format graph ComfyUI's ``POST /prompt`` accepts."""
    return {
        "load_checkpoint": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
            "_meta": {"title": "WAI-illustrious-SDXL"},
        },
        "clip_skip": {
            "class_type": "CLIPSetLastLayer",
            "inputs": {
                "clip": ["load_checkpoint", 1],
                "stop_at_clip_layer": params.clip_skip,
            },
            "_meta": {"title": "CLIP skip"},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip_skip", 0], "text": params.prompt},
            "_meta": {"title": "Prompt"},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["clip_skip", 0], "text": params.negative_prompt},
            "_meta": {"title": "Negative prompt"},
        },
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
                "model": ["load_checkpoint", 0],
                "seed": params.seed,
                "steps": params.steps,
                "cfg": params.cfg,
                "sampler_name": params.sampler_name,
                "scheduler": params.scheduler,
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "latent_image": ["latent", 0],
                "denoise": params.denoise,
            },
            "_meta": {"title": "Sample"},
        },
        "decode": {
            "class_type": "VAEDecode",
            # The VAE comes out of the checkpoint's third output, not a loader.
            "inputs": {"samples": ["sample", 0], "vae": ["load_checkpoint", 2]},
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
