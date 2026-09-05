"""Dark Beast v3's model-specific slice of the shared ComfyUI service."""

from fastapi import FastAPI
from pydantic import Field

import workflow
from comfyui_modal.server import BaseGenerateRequest, ModelService, comfy_client
from comfyui_modal.server import create_app as _create_app
from comfyui_modal.server import run_generation as _run_generation

__all__ = ["GenerateRequest", "comfy_client", "create_app", "run_generation"]


class GenerateRequest(BaseGenerateRequest):
    """Krea 2 sampler parameters, defaulted to the turbo reference settings."""

    prompt: str = Field(description="Natural-language prompt.")
    negative_prompt: str = Field(
        default="",
        description=(
            "Only meaningful alongside a raised cfg: at the default cfg 1 the "
            "negative branch is never consulted."
        ),
    )

    steps: int = Field(default=workflow.DEFAULT_STEPS, ge=1, le=200)
    cfg: float = Field(default=workflow.DEFAULT_CFG, ge=0.0, le=100.0)
    scheduler: str = workflow.DEFAULT_SCHEDULER
    denoise: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "How far the source is re-noised. Only meaningful with source_image; "
            "against an empty latent anything below 1 just underbakes."
        ),
    )

    # Base64 on the way in; the shared layer uploads it to ComfyUI and replaces
    # this with the input-directory filename before `_resolve` runs.
    source_image: str | None = Field(
        default=None,
        description=(
            "Base64 image to start from. Supplying one turns this into img2img: "
            "the output size comes from the source rather than width/height, "
            "batch_size must be 1, and `denoise` finally does something."
        ),
    )
    source_megapixels: float = Field(
        default=1.0,
        ge=0.01,
        le=16.0,
        description="The source is scaled to this budget before encoding. Ignored without one.",
    )

    sampler_name: str = workflow.DEFAULT_SAMPLER
    filename_prefix: str = "darkbeast3"


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
        denoise=request.denoise,
        filename_prefix=request.filename_prefix,
        # A filename by now, not base64 — see `upload_fields`.
        source_image=request.source_image,
        source_megapixels=request.source_megapixels,
    )


SERVICE = ModelService(
    title="Dark Beast v3 (Krea 2) on ComfyUI",
    description="A ComfyUI server with the Dark Beast v3 Krea 2 finetune, plus a typed generate endpoint.",
    request_model=GenerateRequest,
    resolve=_resolve,
    build_workflow=workflow.build_workflow,
    output_node_id=workflow.OUTPUT_NODE_ID,
    upload_fields=("source_image",),
)


def _register_extra_routes(web_app: FastAPI) -> None:
    @web_app.get("/defaults")
    async def defaults() -> dict:
        """The sampler settings this service applies when you omit them."""
        return {
            "diffusion_model": workflow.DIFFUSION_MODEL,
            "text_encoder": workflow.TEXT_ENCODER,
            "vae": workflow.VAE,
            "steps": workflow.DEFAULT_STEPS,
            "cfg": workflow.DEFAULT_CFG,
            "sampler_name": workflow.DEFAULT_SAMPLER,
            "scheduler": workflow.DEFAULT_SCHEDULER,
            # Says outright that these are template values, not the author's.
            "source": workflow.DEFAULTS_SOURCE,
            "aspect_ratios": sorted(workflow.ASPECT_RATIOS),
        }


def create_app(comfy_url: str) -> FastAPI:
    return _create_app(comfy_url, SERVICE, extra_routes=_register_extra_routes)


async def run_generation(client, request: GenerateRequest):
    """Two-argument form, for callers that already know which service this is."""
    return await _run_generation(client, request, SERVICE)
