"""RedCraft v3 (Krea 2) node, rendering on a remote Modal deployment."""

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

ENV_URL = "REDCRAFTV3_MODAL_URL"
CATEGORY = "RedCraft v3 / Krea 2 (Modal)"

# `er_sde` is listed first because the version notes name it first — but `euler`
# is the default, since er_sde is not present in every ComfyUI build and a
# default that fails to resolve would break every render.
SAMPLERS = ["euler", "er_sde", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "ddim"]
SCHEDULERS = ["simple", "beta", "normal", "karras", "sgm_uniform", "exponential"]


class RedCraftV3Modal:
    """Text to image on the remote RedCraft v3 deployment."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a rain-soaked neon alley at night, shot on a handheld camera",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Has no effect at the card's cfg 1; raise cfg to use it.",
                    },
                ),
                **common_geometry_inputs(),
                "steps": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 200,
                        "tooltip": "The version notes publish 8-12; 10 is the midpoint.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "The card specifies 1. Raising it is off-recipe.",
                    },
                ),
                "sampler_name": (
                    SAMPLERS,
                    {
                        "default": "euler",
                        "tooltip": "The card names er_sde and euler interchangeably.",
                    },
                ),
                "scheduler": (SCHEDULERS, {"default": "simple"}),
            },
            "optional": endpoint_inputs(ENV_URL),
            # Lets the progress bar attach to this node rather than the graph.
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "info")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Render with RedCraft v3 (Krea 2) on a Modal-hosted ComfyUI and return the image."

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
            # Give the remote a little slack so it reports the timeout rather
            # than the socket dying underneath us.
            "timeout_s": max(timeout_s - 15.0, 30.0),
            **geometry_payload(aspect_ratio, megapixels, width, height),
        }

        with ProgressMirror(url, client_id, unique_id):
            result = post(url, "/generate", payload, timeout_s)

        params = result.get("params", {})
        notes = ""
        if cfg <= 1.0 and negative_prompt.strip():
            notes = " (negative prompt inactive at cfg 1)"
        info = (
            f"seed={params.get('seed')} steps={params.get('steps')} "
            f"cfg={params.get('cfg')} {params.get('sampler_name')}/{params.get('scheduler')} "
            f"{params.get('width')}x{params.get('height')} "
            f"in {result.get('duration_s')}s{notes}"
        )
        return (to_tensor(result["images"]), int(params.get("seed", seed)), info)


def _endpoint_url(override: str) -> str:
    """The widget arg shadows the imported `endpoint`, hence the indirection."""
    return endpoint(override, ENV_URL)


NODE_CLASS_MAPPINGS = {"RedCraftV3Modal": RedCraftV3Modal}

NODE_DISPLAY_NAME_MAPPINGS = {"RedCraftV3Modal": "RedCraft v3 / Krea 2 (Modal)"}
