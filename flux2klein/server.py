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

    steps: int | None = Field(default=None, ge=1, le=200, description="Overrides the variant.")
    cfg: float | None = Field(default=None, ge=0.0, le=100.0, description="Overrides the variant.")

    filename_prefix: str = "flux2-klein"


def _resolve(request: GenerateRequest) -> workflow.GenerationParams:
    width, height = request.dimensions()
    return workflow.resolve_params(
        request.prompt,
        negative_prompt=request.negative_prompt,
        variant=request.variant,
        width=width,
        height=height,
        seed=request.seed,
        batch_size=request.batch_size,
        steps=request.steps,
        cfg=request.cfg,
        sampler_name=request.sampler_name,
        filename_prefix=request.filename_prefix,
    )


SERVICE = ModelService(
    title="FLUX.2 klein 9B on ComfyUI",
    description="A ComfyUI server with the FLUX.2 klein 9B weights, plus a typed generate endpoint.",
    request_model=GenerateRequest,
    resolve=_resolve,
    build_workflow=workflow.build_workflow,
    output_node_id=workflow.OUTPUT_NODE_ID,
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
