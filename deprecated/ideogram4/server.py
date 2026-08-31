"""Ideogram 4's model-specific slice of the shared ComfyUI service.

Everything generic — the reverse proxy, the submit/poll/fetch loop, the
websocket bridge — lives in `comfyui_modal.server`. This module supplies only
the request shape, how to resolve it, and the two extra routes Ideogram 4 needs.
"""

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import Field

import workflow
from comfyui_modal.server import BaseGenerateRequest, ModelService, comfy_client
from comfyui_modal.server import create_app as _create_app
from comfyui_modal.server import run_generation as _run_generation

__all__ = ["GenerateRequest", "comfy_client", "create_app", "run_generation"]


class GenerateRequest(BaseGenerateRequest):
    """Ideogram 4 parameters, with the preset table applied server-side."""

    prompt: str | None = Field(
        default=None,
        description="Plain-text prompt. Ignored when json_prompt is supplied.",
    )
    json_prompt: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Structured Ideogram 4 caption. Strongly preferred: the model is "
            "trained on this format, and plain text tends to produce an image "
            "with little relation to the prompt."
        ),
    )

    preset: Literal["Quality", "Default", "Turbo"] = workflow.DEFAULT_PRESET
    steps: int | None = Field(default=None, ge=1, le=200)
    mu: float | None = Field(default=None, ge=-10.0, le=10.0)
    std: float | None = Field(default=None, ge=0.1, le=5.0)

    cfg: float = Field(default=workflow.DEFAULT_CFG, ge=0.0, le=100.0)
    late_cfg: float = Field(default=workflow.DEFAULT_LATE_CFG, ge=0.0, le=100.0)
    late_cfg_start: float = Field(default=workflow.DEFAULT_LATE_CFG_START, ge=0.0, le=1.0)

    filename_prefix: str = "ideogram4"

    def caption(self) -> str:
        """The string handed to the text encoder."""
        if self.json_prompt is not None:
            import json

            return json.dumps(self.json_prompt, ensure_ascii=False, indent=4)
        if self.prompt and self.prompt.strip():
            return self.prompt
        raise HTTPException(422, "supply either 'prompt' or 'json_prompt'")


def _resolve(request: GenerateRequest) -> workflow.GenerationParams:
    width, height = request.dimensions()
    return workflow.resolve_params(
        request.caption(),
        width=width,
        height=height,
        seed=request.seed,
        batch_size=request.batch_size,
        preset=request.preset,
        steps=request.steps,
        mu=request.mu,
        std=request.std,
        cfg=request.cfg,
        late_cfg=request.late_cfg,
        late_cfg_start=request.late_cfg_start,
        sampler_name=request.sampler_name,
        filename_prefix=request.filename_prefix,
    )


SERVICE = ModelService(
    title="Ideogram 4 on ComfyUI",
    description="A ComfyUI server with the Ideogram 4 weights, plus a typed generate endpoint.",
    request_model=GenerateRequest,
    resolve=_resolve,
    build_workflow=workflow.build_workflow,
    output_node_id=workflow.OUTPUT_NODE_ID,
)


def _extra_routes(caption_template_path: Path):
    def register(web_app: FastAPI) -> None:
        @web_app.get("/presets")
        async def presets() -> dict:
            return {
                "sampling_presets": workflow.SAMPLING_PRESETS,
                "aspect_ratios": sorted(workflow.ASPECT_RATIOS),
                "resolution": {
                    "min": workflow.MIN_SIDE,
                    "max": workflow.MAX_SIDE,
                    "multiple_of": workflow.SIDE_MULTIPLE,
                },
            }

        @web_app.get("/caption-template")
        async def caption_template(
            prompt: str = "", width: int = 1024, height: int = 1024
        ) -> Response:
            """The magic-prompt template, filled in.

            Ideogram 4 is trained on structured JSON captions. Feed this text to
            any instruction-following LLM and post the result back as
            ``json_prompt``.
            """
            if not caption_template_path.is_file():
                raise HTTPException(404, "caption template not bundled with this deployment")
            text = caption_template_path.read_text(encoding="utf-8")
            filled = (
                text.replace("{{original_prompt}}", prompt)
                .replace("{{width}}", str(workflow.snap_side(width)))
                .replace("{{height}}", str(workflow.snap_side(height)))
            )
            return Response(filled, media_type="text/plain; charset=utf-8")

    return register


def create_app(comfy_url: str, caption_template_path: str | Path) -> FastAPI:
    return _create_app(comfy_url, SERVICE, extra_routes=_extra_routes(Path(caption_template_path)))


async def run_generation(client, request: GenerateRequest):
    """Two-argument form, for callers that already know which service this is."""
    return await _run_generation(client, request, SERVICE)
