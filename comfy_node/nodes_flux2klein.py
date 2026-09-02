"""FLUX.2 klein 9B node, rendering on a remote Modal deployment."""

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

ENV_URL = "FLUX2KLEIN_MODAL_URL"
CATEGORY = "FLUX.2 klein (Modal)"
VARIANTS = ["base", "distilled", "ponpoke-uncensored"]

# "none" rather than an empty string so the dropdown reads clearly. Mirrors the
# service's registry; /variants is the authoritative list.
LORAS = ["none", "snofs-v1.4", "realstockings-v2", "realism-engine-v2"]


class Flux2KleinModal:
    """Text to image on the remote FLUX.2 klein 9B deployment."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a vintage motorcycle parked in front of a retro diner at sunset",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Ignored by the distilled variant, which is guidance-distilled.",
                    },
                ),
                "variant": (
                    VARIANTS,
                    {
                        "default": "base",
                        "tooltip": "base = 20 steps at cfg 5. distilled = 4 steps, ignores cfg.",
                    },
                ),
                **common_geometry_inputs(),
                "override_sampler": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Off means the variant's own steps/cfg are used.",
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                # Appended at the tail: ComfyUI matches widget values by
                # position, so inserting mid-list would shift every value in
                # workflows people have already saved.
                "lora": (LORAS, {"default": "none"}),
                "lora_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
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
    DESCRIPTION = "Render with FLUX.2 klein 9B on a Modal-hosted ComfyUI and return the image."

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        variant: str,
        aspect_ratio: str,
        megapixels: float,
        width: int,
        height: int,
        batch_size: int,
        seed: int,
        override_sampler: bool,
        steps: int,
        cfg: float,
        lora: str = "none",
        lora_strength: float = 1.0,
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
            "variant": variant,
            "batch_size": batch_size,
            "seed": seed,
            # Give the remote a little slack so it reports the timeout rather
            # than the socket dying underneath us.
            "timeout_s": max(timeout_s - 15.0, 30.0),
            **geometry_payload(aspect_ratio, megapixels, width, height),
        }
        # Left out, the server falls back to the variant's tuned steps/cfg.
        if override_sampler:
            payload["steps"] = steps
            payload["cfg"] = cfg
        if lora != "none":
            payload["lora"] = lora
            payload["lora_strength"] = lora_strength

        with ProgressMirror(url, client_id, unique_id):
            result = post(url, "/generate", payload, timeout_s)

        params = result.get("params", {})
        notes = ""
        if variant == "distilled" and negative_prompt.strip():
            notes = " (negative prompt ignored: distilled variant)"
        if params.get("lora"):
            notes += f" lora={params['lora']}@{params.get('lora_strength')}"
        info = (
            f"{params.get('variant')} seed={params.get('seed')} "
            f"steps={params.get('steps')} cfg={params.get('cfg')} "
            f"{params.get('width')}x{params.get('height')} "
            f"in {result.get('duration_s')}s{notes}"
        )
        return (to_tensor(result["images"]), int(params.get("seed", seed)), info)


def _endpoint_url(override: str) -> str:
    """The widget arg shadows the imported `endpoint`, hence the indirection."""
    return endpoint(override, ENV_URL)


NODE_CLASS_MAPPINGS = {"Flux2KleinModal": Flux2KleinModal}

NODE_DISPLAY_NAME_MAPPINGS = {"Flux2KleinModal": "FLUX.2 klein (Modal)"}
