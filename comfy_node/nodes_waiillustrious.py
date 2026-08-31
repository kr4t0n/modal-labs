"""WAI-illustrious-SDXL node, rendering on a remote Modal deployment."""

from __future__ import annotations

import uuid
from typing import Any

from ._runtime import (
    ProgressMirror,
    common_geometry_inputs,
    endpoint,
    endpoint_inputs,
    geometry_payload,
    post,
    to_tensor,
)

ENV_URL = "WAIILLUSTRIOUS_MODAL_URL"
CATEGORY = "WAI-illustrious (Modal)"

SAMPLERS = ["euler_ancestral", "euler", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddim"]
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "beta"]

# The standard Danbooru negative; mirrored from the service so the widget shows
# what the server would apply anyway.
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "normal quality, jpeg artifacts, signature, watermark, username, blurry"
)


class WaiIllustriousModal:
    """Text to image on the remote WAI-illustrious-SDXL deployment."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "1girl, solo, silver hair, red eyes, city at night, masterpiece, best quality",
                        "tooltip": "Danbooru-style tags work best.",
                    },
                ),
                "negative_prompt": ("STRING", {"multiline": True, "default": DEFAULT_NEGATIVE}),
                **common_geometry_inputs(),
                "steps": ("INT", {"default": 28, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (SCHEDULERS, {"default": "normal"}),
                "clip_skip": (
                    "INT",
                    {
                        "default": -2,
                        "min": -24,
                        "max": -1,
                        "tooltip": "-2 is the convention for booru-tagged SDXL finetunes.",
                    },
                ),
            },
            "optional": endpoint_inputs(ENV_URL),
            # Lets the progress bar attach to this node rather than the graph.
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "info")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Render with WAI-illustrious-SDXL on a Modal-hosted ComfyUI and return the image."

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        megapixels: float,
        width: int,
        height: int,
        batch_size: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        clip_skip: int,
        endpoint: str = "",
        timeout_s: float = 900.0,
        unique_id: str | None = None,
    ):
        url = _endpoint_url(endpoint)
        client_id = uuid.uuid4().hex
        payload: dict[str, Any] = {
            "client_id": client_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "batch_size": batch_size,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "clip_skip": clip_skip,
            # Give the remote a little slack so it reports the timeout rather
            # than the socket dying underneath us.
            "timeout_s": max(timeout_s - 15.0, 30.0),
            **geometry_payload(aspect_ratio, megapixels, width, height),
        }

        with ProgressMirror(url, client_id, unique_id):
            result = post(url, "/generate", payload, timeout_s)

        params = result.get("params", {})
        info = (
            f"seed={params.get('seed')} steps={params.get('steps')} "
            f"cfg={params.get('cfg')} {params.get('sampler_name')}/{params.get('scheduler')} "
            f"{params.get('width')}x{params.get('height')} in {result.get('duration_s')}s"
        )
        return (to_tensor(result["images"]), int(params.get("seed", seed)), info)


def _endpoint_url(override: str) -> str:
    """The widget arg shadows the imported `endpoint`, hence the indirection."""
    return endpoint(override, ENV_URL)


NODE_CLASS_MAPPINGS = {"WaiIllustriousModal": WaiIllustriousModal}

NODE_DISPLAY_NAME_MAPPINGS = {"WaiIllustriousModal": "WAI-illustrious (Modal)"}
