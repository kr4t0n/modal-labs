"""Ideogram 4 text-to-image graph, emitted in ComfyUI's ``/prompt`` (API) format.

This is a flattened form of ComfyUI's official ``image_ideogram4_t2i`` template.
The template drives the sampler through a chain of utility nodes (a preset
lookup table, JSON field extraction, dimension snapping). Those are widget
plumbing rather than model nodes, so they are evaluated here in Python and the
emitted graph contains only the nodes that touch weights.

Graph shape (all nodes are ComfyUI core, no custom node packs)::

    UNETLoader(cond) -> CFGOverride ----------.
    UNETLoader(uncond) ----------------------. |
    CLIPLoader -> CLIPTextEncode -> ConditioningZeroOut
                        |            |       | |
                        `-> DualModelGuider <-' '
                                 |
    RandomNoise / KSamplerSelect / Ideogram4Scheduler / EmptyFlux2LatentImage
                                 |
                        SamplerCustomAdvanced -> VAEDecode -> SaveImage
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

# --- Weight files -----------------------------------------------------------
# Every file lives in the Comfy-Org/Ideogram-4 repo; see app.py for the mirror.
DIFFUSION_MODEL = "ideogram4_fp8_scaled.safetensors"
UNCONDITIONAL_DIFFUSION_MODEL = "ideogram4_unconditional_fp8_scaled.safetensors"
TEXT_ENCODER = "qwen3vl_8b_fp8_scaled.safetensors"
VAE = "flux2-vae.safetensors"

# --- Sampling ---------------------------------------------------------------
# Ideogram 4 ships three reference schedules. `mu`/`std` parameterise the
# logit-normal noise schedule that Ideogram4Scheduler builds; they are not
# interchangeable with the shift value used by Flux-style schedulers.
SAMPLING_PRESETS: dict[str, dict[str, float]] = {
    "Quality": {"steps": 48, "mu": 0.0, "std": 1.5},
    "Default": {"steps": 20, "mu": 0.0, "std": 1.75},
    "Turbo": {"steps": 12, "mu": 0.5, "std": 1.75},
}
DEFAULT_PRESET = "Default"

# Dual-branch CFG: `cfg` steers prompt adherence for the whole run, then
# `late_cfg` takes over from `late_cfg_start` onwards to keep late steps from
# over-sharpening. Values match the shipped template.
DEFAULT_CFG = 7.0
DEFAULT_LATE_CFG = 3.0
DEFAULT_LATE_CFG_START = 0.7


@dataclass(frozen=True)
class GenerationParams:
    """Fully resolved sampler settings — no defaults left to the graph."""

    prompt: str
    width: int
    height: int
    seed: int
    batch_size: int
    steps: int
    mu: float
    std: float
    cfg: float
    late_cfg: float
    late_cfg_start: float
    sampler_name: str
    filename_prefix: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_params(
    prompt: str,
    *,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    batch_size: int = 1,
    preset: str = DEFAULT_PRESET,
    steps: int | None = None,
    mu: float | None = None,
    std: float | None = None,
    cfg: float = DEFAULT_CFG,
    late_cfg: float = DEFAULT_LATE_CFG,
    late_cfg_start: float = DEFAULT_LATE_CFG_START,
    sampler_name: str = "euler",
    filename_prefix: str = "ideogram4",
) -> GenerationParams:
    """Apply the preset table and snap dimensions; explicit values win."""
    if not prompt or not prompt.strip():
        raise WorkflowError("prompt must not be empty")
    if preset not in SAMPLING_PRESETS:
        raise WorkflowError(
            f"unknown preset {preset!r}; expected one of {sorted(SAMPLING_PRESETS)}"
        )
    if batch_size < 1:
        raise WorkflowError("batch_size must be at least 1")

    defaults = SAMPLING_PRESETS[preset]
    resolved_steps = int(steps if steps is not None else defaults["steps"])
    if resolved_steps < 1:
        raise WorkflowError("steps must be at least 1")

    return GenerationParams(
        prompt=prompt,
        width=snap_side(width),
        height=snap_side(height),
        seed=normalise_seed(seed),
        batch_size=int(batch_size),
        steps=resolved_steps,
        mu=float(mu if mu is not None else defaults["mu"]),
        std=float(std if std is not None else defaults["std"]),
        cfg=float(cfg),
        late_cfg=float(late_cfg),
        late_cfg_start=float(late_cfg_start),
        sampler_name=sampler_name,
        filename_prefix=filename_prefix,
    )


# The SaveImage node id, so callers know which history entry holds the results.
OUTPUT_NODE_ID = "save_image"


def build_workflow(params: GenerationParams) -> dict[str, Any]:
    """Emit the API-format graph ComfyUI's ``POST /prompt`` accepts."""
    return {
        "load_unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
            "_meta": {"title": "Ideogram 4 (conditional)"},
        },
        # The late-CFG override wraps the conditional model only; the guider
        # reads guider.cfg through it, so it must sit between loader and guider.
        "late_cfg": {
            "class_type": "CFGOverride",
            "inputs": {
                "model": ["load_unet", 0],
                "cfg": params.late_cfg,
                "start_percent": params.late_cfg_start,
                "end_percent": 1.0,
            },
            "_meta": {"title": "Late-step CFG override"},
        },
        "load_unet_uncond": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": UNCONDITIONAL_DIFFUSION_MODEL,
                "weight_dtype": "default",
            },
            "_meta": {"title": "Ideogram 4 (unconditional)"},
        },
        "load_clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER,
                "type": "ideogram4",
                "device": "default",
            },
            "_meta": {"title": "Qwen3-VL 8B text encoder"},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["load_clip", 0], "text": params.prompt},
            "_meta": {"title": "Prompt"},
        },
        "negative": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["positive", 0]},
            "_meta": {"title": "Zeroed conditioning"},
        },
        "guider": {
            "class_type": "DualModelGuider",
            "inputs": {
                "model": ["late_cfg", 0],
                "positive": ["positive", 0],
                "cfg": params.cfg,
                "model_negative": ["load_unet_uncond", 0],
                "negative": ["negative", 0],
            },
            "_meta": {"title": "Dual-branch CFG"},
        },
        "sampler": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": params.sampler_name},
            "_meta": {"title": "Sampler"},
        },
        "sigmas": {
            "class_type": "Ideogram4Scheduler",
            "inputs": {
                "steps": params.steps,
                "width": params.width,
                "height": params.height,
                "mu": params.mu,
                "std": params.std,
            },
            "_meta": {"title": "Ideogram 4 schedule"},
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
