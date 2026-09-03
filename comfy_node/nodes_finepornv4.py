"""FinePorn v4 (Krea 2) node, rendering on a remote Modal deployment."""

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

ENV_URL = "FINEPORNV4_MODAL_URL"
CATEGORY = "FinePorn v4 / Krea 2 (Modal)"

# `er_sde` leads: the model card names it for v4 alongside euler, and it is the
# pairing its author calls fast and stable. `res_2s` is listed for v1 only and
# is absent from some ComfyUI builds, so it is deliberately not offered here.
SAMPLERS = ["euler", "er_sde", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "ddim"]
SCHEDULERS = ["beta", "simple", "normal", "karras", "sgm_uniform", "exponential"]

# The card reports standard Krea 2 resolutions underperform on this merge and
# recommends scaling them by 1.25x. Mirrors DEFAULT_SIDE/DEFAULT_MEGAPIXELS in
# the deployment's workflow.py; /defaults is the authoritative copy.
DEFAULT_SIDE = 1280
DEFAULT_MEGAPIXELS = 1.64


class FinePornV4Modal:
    """Text to image on the remote FinePorn v4 deployment."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "this is an amateur photo taken from smartphone, casual photo "
                            "of a woman laughing in a sunlit kitchen"
                        ),
                        "tooltip": (
                            "The merge targets a smartphone-snapshot look. Its author "
                            "recommends opening with 'this is a casual, low-quality photo' "
                            "or similar; without it results read flatter."
                        ),
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Has no effect at the default cfg 1; raise cfg to use it.",
                    },
                ),
                **common_geometry_inputs(
                    default_side=DEFAULT_SIDE, default_megapixels=DEFAULT_MEGAPIXELS
                ),
                "steps": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 200,
                        "tooltip": "The card publishes 8-12 across versions; 10 is the midpoint.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "A turbo merge; it samples at 1. Raising it is off-recipe.",
                    },
                ),
                "sampler_name": (SAMPLERS, {"default": "euler"}),
                "scheduler": (SCHEDULERS, {"default": "beta"}),
            },
            "optional": endpoint_inputs(ENV_URL),
            # Lets the progress bar attach to this node rather than the graph.
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "info")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Render with FinePorn v4 (Krea 2) on a Modal-hosted ComfyUI and return the image."

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


NODE_CLASS_MAPPINGS = {"FinePornV4Modal": FinePornV4Modal}

NODE_DISPLAY_NAME_MAPPINGS = {"FinePornV4Modal": "FinePorn v4 / Krea 2 (Modal)"}
