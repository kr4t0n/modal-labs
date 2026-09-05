"""FLUX.2 klein's model-specific slice of the shared ComfyUI service.

Everything generic — the reverse proxy, the submit/poll/fetch loop, the
websocket bridge — lives in `comfyui_modal.server`. This module supplies only
the request shape, how to resolve it, and the `/variants` route.
"""

from typing import Literal

from fastapi import FastAPI
from pydantic import Field

import workflow
from comfyui_modal.server import BaseGenerateRequest, ModelService, comfy_client
from comfyui_modal.server import create_app as _create_app
from comfyui_modal.server import run_generation as _run_generation

__all__ = ["GenerateRequest", "comfy_client", "create_app", "run_generation"]


class GenerateRequest(BaseGenerateRequest):
    """FLUX.2 klein parameters, with the variant defaults applied server-side."""

    prompt: str = Field(description="Natural-language prompt.")
    negative_prompt: str = Field(
        default="",
        description="Only meaningful on the base variant; the distilled one ignores it.",
    )
    variant: Literal["base", "distilled", "ponpoke-uncensored"] = Field(
        default=workflow.DEFAULT_VARIANT,
        description=(
            "'base' is undistilled (20 steps, cfg 5). 'distilled' is a 4-step "
            "guidance-distilled model that ignores cfg and negative prompts."
        ),
    )

    lora: str | None = Field(
        default=None,
        description=(
            "Name of an adapter to layer onto the transformer. See /loras; "
            "omit for the plain variant."
        ),
    )
    lora_strength: float | None = Field(
        default=None,
        ge=-10.0,
        le=10.0,
        description=(
            "Omit to use the adapter's own recommended strength, which /variants "
            "reports per adapter. Ignored when no lora is named."
        ),
    )

    steps: int | None = Field(default=None, ge=1, le=200, description="Overrides the variant.")
    cfg: float | None = Field(default=None, ge=0.0, le=100.0, description="Overrides the variant.")

    filename_prefix: str = "flux2-klein"

    # Base64 on the way in; the shared layer uploads each to ComfyUI and
    # replaces this with the input-directory filenames before `_resolve` runs.
    # Declared in `upload_fields` below, which is what performs that swap.
    reference_images: list[str] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Base64-encoded images to edit from. Supplying any turns this into "
            "an image edit: the output size is taken from the first reference "
            "rather than width/height, and batch_size must be 1. Not supported "
            "on the distilled variant."
        ),
    )
    reference_megapixels: float = Field(
        default=1.0,
        ge=0.01,
        le=16.0,
        description="Each reference is scaled to this budget before encoding. Ignored with none.",
    )


def _resolve(request: GenerateRequest) -> workflow.GenerationParams:
    width, height = request.dimensions()
    return workflow.resolve_params(
        request.prompt,
        negative_prompt=request.negative_prompt,
        variant=request.variant,
        lora=request.lora,
        lora_strength=request.lora_strength,
        width=width,
        height=height,
        seed=request.seed,
        batch_size=request.batch_size,
        steps=request.steps,
        cfg=request.cfg,
        sampler_name=request.sampler_name,
        filename_prefix=request.filename_prefix,
        # Filenames by now, not base64 — see `upload_fields`.
        reference_images=request.reference_images,
        reference_megapixels=request.reference_megapixels,
    )


SERVICE = ModelService(
    title="FLUX.2 klein 9B on ComfyUI",
    description="A ComfyUI server with the FLUX.2 klein 9B weights, plus a typed generate endpoint.",
    request_model=GenerateRequest,
    resolve=_resolve,
    build_workflow=workflow.build_workflow,
    output_node_id=workflow.OUTPUT_NODE_ID,
    upload_fields=("reference_images",),
)


def _register_extra_routes(web_app: FastAPI) -> None:
    @web_app.get("/variants")
    async def variants() -> dict:
        return {
            "variants": {
                name: {
                    "checkpoint": spec.checkpoint,
                    "steps": spec.steps,
                    "cfg": spec.cfg,
                    "description": spec.description,
                }
                for name, spec in workflow.VARIANTS.items()
            },
            "default": workflow.DEFAULT_VARIANT,
            "loras": {
                name: {
                    "filename": spec.filename,
                    "description": spec.description,
                    "trained_on": spec.trained_on,
                    "trigger_words": list(spec.trigger_words),
                    # The published band, and the single value applied when the
                    # request omits lora_strength.
                    "recommended_strength": (
                        list(spec.recommended_strength) if spec.recommended_strength else None
                    ),
                    "default_strength": spec.default_strength,
                }
                for name, spec in workflow.LORAS.items()
            },
            "default_lora_strength": workflow.DEFAULT_LORA_STRENGTH,
            "aspect_ratios": sorted(workflow.ASPECT_RATIOS),
            "resolution": {
                "min": workflow.MIN_SIDE,
                "max": workflow.MAX_SIDE,
                "multiple_of": workflow.SIDE_MULTIPLE,
            },
        }


def create_app(comfy_url: str) -> FastAPI:
    return _create_app(comfy_url, SERVICE, extra_routes=_register_extra_routes)


async def run_generation(client, request: GenerateRequest):
    """Two-argument form, for callers that already know which service this is."""
    return await _run_generation(client, request, SERVICE)
