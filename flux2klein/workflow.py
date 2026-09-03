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
# Everything except the VAE is gated on Hugging Face; see app.py for the token.
TEXT_ENCODER = "qwen_3_8b_fp8mixed.safetensors"

# An abliterated Qwen3-8B: the refusal direction is orthogonalised out of the
# encoder's mid/late layers. Same architecture, so ComfyUI's `flux2` CLIP type
# loads it by the same path — but bf16 rather than fp8mixed, so it is roughly
# twice the size on disk and in VRAM.
UNCENSORED_TEXT_ENCODER = "qwen_3_8b_uncensored_bf16.safetensors"
# Not flux2-vae: the klein templates ship the small-decoder autoencoder.
VAE = "full_encoder_small_decoder.safetensors"

# The 9B text encoder pairs with the 9B transformer. Mixing in the 4B encoder
# degrades quality rather than failing loudly, so the pairing is fixed here.
CLIP_TYPE = "flux2"


@dataclass(frozen=True)
class Variant:
    """A checkpoint, its text encoder, and the sampler settings it was tuned for.

    The encoder is part of the variant rather than a global: swapping it changes
    what the model will render, so it should not be settable independently of
    the checkpoint it was validated against.
    """

    checkpoint: str
    steps: int
    cfg: float
    description: str
    text_encoder: str = TEXT_ENCODER


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
    # Named for the encoder's author, since that is the only thing that differs:
    # same transformer and schedule as `base`, so any change in output comes
    # from prompt handling alone.
    "ponpoke-uncensored": Variant(
        checkpoint="flux-2-klein-base-9b-fp8.safetensors",
        steps=20,
        cfg=5.0,
        description=(
            "As base, with an abliterated text encoder: prompt-stage refusals "
            "are removed. bf16 encoder, so ~8 GB more VRAM than base."
        ),
        text_encoder=UNCENSORED_TEXT_ENCODER,
    ),
}
DEFAULT_VARIANT = "base"


# The fallback for adapters whose author publishes no guidance; the
# conventional full-strength value. Adapters that *do* publish a range carry it
# on the registry entry and are resolved per-adapter, not from this.
DEFAULT_LORA_STRENGTH = 1.0


@dataclass(frozen=True)
class Lora:
    """An adapter that can be layered onto the klein transformer."""

    filename: str
    description: str
    # Adapters are trained against one transformer; applying to another is not
    # an error, it just may not behave as trained.
    trained_on: str = "base"
    # Words the adapter was trained to respond to. Omitting them from a prompt
    # is the usual reason a LoRA "does nothing" — it loads fine and simply is
    # not invoked, so the registry carries them where callers can see them.
    trigger_words: tuple[str, ...] = ()
    # The strength range the author publishes, when they publish one. Stored as
    # the range rather than a single number because that is what model cards
    # actually say, and the spread is information a caller may want to explore.
    recommended_strength: tuple[float, float] | None = None

    @property
    def default_strength(self) -> float:
        """The strength to apply when the caller names none.

        The midpoint of the published range. Not the low end: these ranges are
        the author's working band rather than a safety limit, so the middle is
        the least surprising pick. Adapters with no published range fall back to
        the service default, which is what every adapter used before this field
        existed.
        """
        if self.recommended_strength is None:
            return DEFAULT_LORA_STRENGTH
        low, high = self.recommended_strength
        return round((low + high) / 2, 3)


# A registry rather than a free-form filename: the weights have to be fetched
# ahead of time, so the set of usable adapters is known at deploy time. Adding
# one means an entry here plus a ModelFile in app.py.
LORAS: dict[str, Lora] = {
    "snofs-v1.4": Lora(
        filename="klein_snofs_v1_4.safetensors",
        description="Ashen3 SNOFS v1.4 — a LoKr adapter trained on klein 9B.",
    ),
    "realstockings-v2": Lora(
        filename="RealStockingsV2.safetensors",
        description=(
            "lajmar Stockings v2 (Lace Black & Tan) — a standard LoRA trained "
            "on klein 9B with ai-toolkit."
        ),
        trigger_words=("stockings", "RealStockings"),
    ),
    # No trigger words by design: upstream publishes none, so this one is a
    # general finetune that applies to every prompt rather than a concept
    # summoned by a token. The empty tuple here is a fact, not an omission.
    "realism-engine-v2": Lora(
        filename="Realism_Engine_Klein_V2.safetensors",
        description=(
            "Realism Engine Klein v2 — a general nudity and anatomy finetune "
            "for klein 9B. Adult content. Upstream recommends strength "
            "1.0-1.25, above this service's 1.0 fallback."
        ),
        recommended_strength=(1.0, 1.25),
    ),
    # Upstream's prose and its machine-readable trained-words list disagree:
    # the model card also names "big round boobs" and "topless". The API list
    # is what is pinned here, being the field Civitai actually maintains.
    "nsfw-unlocked-v2": Lora(
        filename="FLUX2_KLEIN_UNLOCKED_V2.safetensors",
        description=(
            "NSFW Unlocked v2, klein build — an explicit-content LoRA trained "
            "on klein 9B. Adult content. Upstream recommends strength 0.5-0.9, "
            "*below* this service's 1.0 fallback, plus 20+ steps at cfg 3.5."
        ),
        trigger_words=("nude", "naked", "blow job", "cum", "ass", "pussy"),
        recommended_strength=(0.5, 0.9),
    ),
    # Civitai's trained-words list is empty for this one, but the model card
    # documents the triggers in prose. Trusting the empty field would register
    # an adapter that loads and silently does nothing, so the prose wins.
    "naturalbeauty-v2": Lora(
        filename="NaturalBeautyFLUX2Klein9BNudity_v2.safetensors",
        description=(
            "NaturalBeauty v2, klein build — a photorealistic female-nudity "
            "LoRA trained on klein 9B. Adult content. Responds to further "
            "tags for pose, hair, body direction and pubic hair; see the "
            "model card."
        ),
        trigger_words=("naked", "topless", "bottomless"),
    ),
}


@dataclass(frozen=True)
class GenerationParams:
    """Fully resolved sampler settings — no defaults left to the graph."""

    prompt: str
    negative_prompt: str
    variant: str
    checkpoint: str
    text_encoder: str
    lora: str | None
    lora_strength: float
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
    lora: str | None = None,
    # None means "no opinion", which is what lets the adapter's own recommended
    # strength apply. An explicit float always wins, including an explicit 1.0.
    lora_strength: float | None = None,
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
    if lora is not None and lora not in LORAS:
        raise WorkflowError(f"unknown lora {lora!r}; expected one of {sorted(LORAS)}")

    spec = VARIANTS[variant]
    resolved_steps = int(steps if steps is not None else spec.steps)
    if resolved_steps < 1:
        raise WorkflowError("steps must be at least 1")

    # Resolved from the adapter rather than a global constant: two of these
    # publish a strength band, and one of them sits below the old 1.0 default,
    # so taking the constant would have overdriven it.
    if lora_strength is not None:
        resolved_strength = float(lora_strength)
    elif lora is not None:
        resolved_strength = LORAS[lora].default_strength
    else:
        resolved_strength = DEFAULT_LORA_STRENGTH

    return GenerationParams(
        prompt=prompt,
        negative_prompt=negative_prompt,
        variant=variant,
        checkpoint=spec.checkpoint,
        text_encoder=spec.text_encoder,
        lora=lora,
        lora_strength=resolved_strength,
        width=snap_side(width),
        height=snap_side(height),
        seed=normalise_seed(seed),
        batch_size=int(batch_size),
        steps=resolved_steps,
        cfg=float(cfg if cfg is not None else spec.cfg),
        sampler_name=sampler_name,
        filename_prefix=filename_prefix,
    )


OUTPUT_NODE_ID = "save_image"


LORA_NODE_ID = "load_lora"


def build_workflow(params: GenerationParams) -> dict[str, Any]:
    """Emit the API-format graph ComfyUI's ``POST /prompt`` accepts.

    When an adapter is requested a `LoraLoaderModelOnly` is spliced between the
    transformer and the guider. Model-only is deliberate: these adapters patch
    `diffusion_model.*` alone, and the text encoder is loaded separately here.
    """
    # Without a LoRA the graph must be byte-identical to what it emitted before
    # the feature existed; the committed reference graph pins that.
    model_source = ["load_unet", 0] if params.lora is None else [LORA_NODE_ID, 0]

    graph: dict[str, Any] = {
        "load_unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": params.checkpoint, "weight_dtype": "default"},
            "_meta": {"title": f"FLUX.2 klein 9B ({params.variant})"},
        },
        "load_clip": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": params.text_encoder,
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
                "model": model_source,
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

    if params.lora is not None:
        graph[LORA_NODE_ID] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["load_unet", 0],
                "lora_name": LORAS[params.lora].filename,
                "strength_model": params.lora_strength,
            },
            "_meta": {"title": f"LoRA: {params.lora}"},
        }

    return graph
