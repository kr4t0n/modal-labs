"""WAI-illustrious-SDXL's model-specific slice of the shared ComfyUI service.

Everything generic — the reverse proxy, the submit/poll/fetch loop, the
websocket bridge — lives in `comfyui_modal.server`. This module supplies only
the request shape, how to resolve it, and the `/defaults` route.
"""

from fastapi import FastAPI
from pydantic import Field

import workflow
from comfyui_modal.server import BaseGenerateRequest, ModelService, comfy_client
from comfyui_modal.server import create_app as _create_app
from comfyui_modal.server import run_generation as _run_generation

__all__ = ["GenerateRequest", "comfy_client", "create_app", "run_generation"]


class GenerateRequest(BaseGenerateRequest):
    """SDXL sampler parameters, with anime-finetune conventions as defaults."""

    prompt: str = Field(description="Danbooru-style tags work best, e.g. '1girl, solo, ...'.")
    negative_prompt: str = Field(
        default=workflow.DEFAULT_NEGATIVE_PROMPT,
        description="Defaults to the standard Danbooru negative; pass '' to opt out.",
    )

    steps: int = Field(default=workflow.DEFAULT_STEPS, ge=1, le=200)
    cfg: float = Field(default=workflow.DEFAULT_CFG, ge=0.0, le=100.0)
    scheduler: str = workflow.DEFAULT_SCHEDULER
    clip_skip: int = Field(
        default=workflow.DEFAULT_CLIP_SKIP,
        ge=-24,
        le=-1,
        description="-2 is the convention for booru-tagged SDXL finetunes.",
    )
    denoise: float = Field(default=1.0, ge=0.0, le=1.0)

    sampler_name: str = workflow.DEFAULT_SAMPLER
    filename_prefix: str = "wai-illustrious"


def _resolve(request: GenerateRequest) -> workflow.GenerationParams:
    width, height = request.dimensions()
    return workflow.resolve_params(
        request.prompt,
        negative_prompt=request.negative_prompt,
        width=width,
        height=height,
        seed=request.seed,
        batch_size=request.batch_size,
        steps=request.steps,
        cfg=request.cfg,
        sampler_name=request.sampler_name,
        scheduler=request.scheduler,
        clip_skip=request.clip_skip,
        denoise=request.denoise,
        filename_prefix=request.filename_prefix,
    )


SERVICE = ModelService(
    title="WAI-illustrious-SDXL on ComfyUI",
    description="A ComfyUI server with the WAI-illustrious-SDXL checkpoint, plus a typed generate endpoint.",
    request_model=GenerateRequest,
    resolve=_resolve,
    build_workflow=workflow.build_workflow,
    output_node_id=workflow.OUTPUT_NODE_ID,
)


def _register_extra_routes(web_app: FastAPI) -> None:
    @web_app.get("/defaults")
    async def defaults() -> dict:
        """The sampler conventions this service applies when you omit them."""
        return {
            "checkpoint": workflow.CHECKPOINT,
            "steps": workflow.DEFAULT_STEPS,
            "cfg": workflow.DEFAULT_CFG,
            "sampler_name": workflow.DEFAULT_SAMPLER,
            "scheduler": workflow.DEFAULT_SCHEDULER,
            "clip_skip": workflow.DEFAULT_CLIP_SKIP,
            "negative_prompt": workflow.DEFAULT_NEGATIVE_PROMPT,
            "native_megapixels": workflow.NATIVE_MEGAPIXELS,
            "aspect_ratios": sorted(workflow.ASPECT_RATIOS),
        }


def create_app(comfy_url: str) -> FastAPI:
    return _create_app(comfy_url, SERVICE, extra_routes=_register_extra_routes)


async def run_generation(client, request: GenerateRequest):
    """Two-argument form, for callers that already know which service this is."""
    return await _run_generation(client, request, SERVICE)
