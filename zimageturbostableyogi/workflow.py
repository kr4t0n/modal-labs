"""Z-Image Turbo (Stable Yogi) text-to-image graph, in ComfyUI's ``/prompt`` format.

This is a community finetune/merge, **not** Alibaba's stock Z-Image Turbo. The
graph is flattened from ComfyUI's official ``image_z_image_turbo`` template with
the stock diffusion model swapped for Stable Yogi's build; the encoder, VAE and
sampler settings are the stock ones. That finetune is
distributed as a *diffusion model only* — its safetensors carries the Lumina-2
style blocks and nothing else — so the Qwen3-4B encoder and the autoencoder are
loaded separately.

Graph shape (all nodes are ComfyUI core)::

    UNETLoader -> ModelSamplingAuraFlow ---------------------.
    CLIPLoader -> CLIPTextEncode (positive) -----------------+-> KSampler
                            `-> ConditioningZeroOut ---------'      |
    EmptySD3LatentImage -----------------------------------'        |
                                                    VAEDecode -> SaveImage

Three details here are easy to get wrong and come from the template rather than
inference: the `ModelSamplingAuraFlow` shift patch is required, the latent is
`EmptySD3LatentImage` (16-channel, /8) rather than the SD-style one, and the
reference sampler is `res_multistep`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from comfyui_modal import graph as graph_fragments
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
DIFFUSION_MODEL = "zimageturbostableyogi.safetensors"
# Z-Image conditions on Qwen3-4B. ComfyUI reaches its encoder through the
# `lumina2` CLIP type — Z-Image's model class subclasses Lumina2 — and the 8B
# encoders the klein services use are not interchangeable.
TEXT_ENCODER = "qwen_3_4b_fp8_mixed.safetensors"
CLIP_TYPE = "lumina2"
VAE = "ae.safetensors"

# --- Sampling ---------------------------------------------------------------
# From ComfyUI's official Z-Image Turbo template, not inferred.
DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "res_multistep"
DEFAULT_SCHEDULER = "simple"
# ModelSamplingAuraFlow's own default is 1.73; Z-Image's model class declares a
# sampling shift of 3.0 and the template patches it to match. Leaving it at the
# node default silently changes the noise schedule.
DEFAULT_SHIFT = 3.0


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
    shift: float
    sampler_name: str
    scheduler: str
    denoise: float
    filename_prefix: str
    # A ComfyUI input-directory filename, already uploaded. Set makes this
    # img2img: the sampler starts from this image instead of an empty latent.
    source_image: str | None = None
    source_megapixels: float = 1.0

    @property
    def is_img2img(self) -> bool:
        return self.source_image is not None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_img2img"] = self.is_img2img
        if self.is_img2img:
            # The encoded source defines the latent, so these were never
            # applied. Reporting them anyway would make `params` a lie.
            data["width"] = data["height"] = None
        return data


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
    shift: float = DEFAULT_SHIFT,
    sampler_name: str = DEFAULT_SAMPLER,
    scheduler: str = DEFAULT_SCHEDULER,
    denoise: float = 1.0,
    source_image: str | None = None,
    source_megapixels: float = 1.0,
    filename_prefix: str = "zimageturbostableyogi",
) -> GenerationParams:
    """Validate and snap the request into a fully specified parameter set."""
    if not prompt or not prompt.strip():
        raise WorkflowError("prompt must not be empty")
    if batch_size < 1:
        raise WorkflowError("batch_size must be at least 1")
    if steps < 1:
        raise WorkflowError("steps must be at least 1")
    if shift <= 0:
        raise WorkflowError("shift must be positive")
    if not 0.0 <= denoise <= 1.0:
        raise WorkflowError("denoise must be between 0 and 1")
    if not 0.01 <= source_megapixels <= 16.0:
        raise WorkflowError("source_megapixels must be between 0.01 and 16")
    if source_image is not None and batch_size != 1:
        # One encoded source is one starting latent, so a batch would be N
        # copies of the same img2img. Refuse rather than surprise.
        raise WorkflowError("batch_size must be 1 for img2img; the source fixes the latent")

    return GenerationParams(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=snap_side(width),
        height=snap_side(height),
        seed=normalise_seed(seed),
        batch_size=int(batch_size),
        steps=int(steps),
        cfg=float(cfg),
        shift=float(shift),
        sampler_name=sampler_name,
        scheduler=scheduler,
        denoise=float(denoise),
        filename_prefix=filename_prefix,
        source_image=source_image,
        source_megapixels=float(source_megapixels),
    )


OUTPUT_NODE_ID = "save_image"
NEGATIVE_NODE_ID = "negative"


def build_workflow(params: GenerationParams) -> dict[str, Any]:
    """Emit the API-format graph ComfyUI's ``POST /prompt`` accepts."""
    graph: dict[str, Any] = {
        "load_unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
            "_meta": {"title": "Z-Image Turbo (Stable Yogi)"},
        },
        "model_sampling": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["load_unet", 0], "shift": params.shift},
            "_meta": {"title": "Sampling shift"},
        },
        "load_clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER,
                "type": CLIP_TYPE,
                "device": "default",
            },
            "_meta": {"title": "Qwen3 4B text encoder"},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["load_clip", 0], "text": params.prompt},
            "_meta": {"title": "Prompt"},
        },
        NEGATIVE_NODE_ID: _negative_node(params),
        "latent": {
            "class_type": "EmptySD3LatentImage",
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
                # The patched model, not the raw loader.
                "model": ["model_sampling", 0],
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

    if params.is_img2img:
        graph_fragments.splice_img2img_source(
            graph,
            filename=params.source_image,
            megapixels=params.source_megapixels,
        )

    return graph


def _negative_node(params: GenerationParams) -> dict[str, Any]:
    """Zeroed conditioning by default, a real encode when text is supplied.

    The reference template zeroes it, because turbo samples at cfg 1 where the
    negative branch is never consulted. Encoding real text there is only
    meaningful alongside a raised cfg, so it is opt-in.
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
