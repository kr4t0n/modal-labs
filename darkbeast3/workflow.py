"""Dark Beast v3 (Krea 2) text-to-image graph, in ComfyUI's ``/prompt`` format.

The fifth Krea 2 service here, and structurally the same as the other four: a
diffusion model only — no text encoder, no VAE — loaded through `UNETLoader`
beside the Qwen3-VL-4B encoder and the Qwen-Image autoencoder.

Graph shape, the Krea 2 turbo reference template::

    UNETLoader ---------------------------------------------.
    CLIPLoader -> CLIPTextEncode (positive) -----------------+-> KSampler
                            `-> ConditioningZeroOut ---------'      |
    EmptyLatentImage --------------------------------------'       |
                                                    VAEDecode -> SaveImage

The upstream listing is titled "H3 Director Edition" and its model description
describes a *video* pipeline — 2K short films, frame interpolation, 6-10 step
sampling. None of that applies here: H3 is a separate version on that page with
MiniMax H3 as its base. This is the Krea 2 still-image build. See AGENTS.md.
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
# Renamed on the way into the Volume. Upstream gives all five precisions of this
# version the name `darkBeastH3Director_darkBeast330.safetensors` — a name that
# says neither which precision it is nor that it is the Krea 2 build rather than
# the H3 one. The destination in app.py is what makes the graph unambiguous.
DIFFUSION_MODEL = "darkbeast_v3_krea2_int8.safetensors"
# Shared verbatim with the ultra, finepornv4, redgpt2gpt and redcraft3
# services: same base model, same companions. A test asserts they stay in step.
TEXT_ENCODER = "qwen3vl_4b_fp8_scaled.safetensors"
CLIP_TYPE = "krea2"
VAE = "qwen_image_vae.safetensors"

# --- Sampling ---------------------------------------------------------------
# ComfyUI's official Krea 2 turbo template values. This version's notes are
# marketing copy and publish no sampler settings, so the template is the honest
# fallback — the same choice ultra and redgpt2gpt make.
#
# The model description does mention "6-10 steps", but that belongs to the H3
# *video* edition on the same page and does not transfer to a still-image Krea 2
# build. Adopting it here would be importing a number from a different model.
DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"

# Reported through /defaults so a caller can tell a template fallback from a
# published recipe without reading the source. redcraft3, by the same author
# over the same base, does publish one and says so there.
DEFAULTS_SOURCE = "ComfyUI's official Krea 2 turbo template; this version publishes no settings"


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
    sampler_name: str = DEFAULT_SAMPLER,
    scheduler: str = DEFAULT_SCHEDULER,
    denoise: float = 1.0,
    source_image: str | None = None,
    source_megapixels: float = 1.0,
    filename_prefix: str = "darkbeast3",
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
            "_meta": {"title": "Dark Beast v3 (Krea 2)"},
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
    if params.is_img2img:
        graph_fragments.splice_img2img_source(
            graph,
            filename=params.source_image,
            megapixels=params.source_megapixels,
        )

    return graph


def _negative_node(params: GenerationParams) -> dict[str, Any]:
    """Zeroed conditioning by default, a real encode when text is supplied.

    Same reasoning as the other Krea 2 services: turbo samples at cfg 1, where
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
