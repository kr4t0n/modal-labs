"""Ideogram 4 nodes, rendering on a remote Modal deployment."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ._runtime import (
    ProgressMirror,
    common_geometry_inputs,
    endpoint,
    endpoint_inputs,
    geometry_payload,
    get,
    post,
    to_tensor,
)

ENV_URL = "IDEOGRAM4_MODAL_URL"
CATEGORY = "Ideogram 4 (Modal)"
PRESETS = ["Default", "Quality", "Turbo"]


class Ideogram4Modal:
    """Text to image on the remote deployment."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a vintage travel poster for the rings of Saturn, bold type reading 'SATURN'",
                        "tooltip": "Plain text, or a structured Ideogram 4 JSON caption.",
                    },
                ),
                "preset": (PRESETS, {"default": "Default"}),
                **common_geometry_inputs(),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": endpoint_inputs(ENV_URL),
            # Lets the progress bar attach to this node rather than the graph.
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "info")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Render with Ideogram 4 on a Modal-hosted ComfyUI and return the image."

    def generate(
        self,
        prompt: str,
        preset: str,
        aspect_ratio: str,
        megapixels: float,
        width: int,
        height: int,
        batch_size: int,
        seed: int,
        cfg: float,
        endpoint: str = "",
        timeout_s: float = 900.0,
        unique_id: str | None = None,
    ):
        url = _endpoint_url(endpoint)
        client_id = uuid.uuid4().hex
        payload: dict[str, Any] = {
            "client_id": client_id,
            "preset": preset,
            "batch_size": batch_size,
            "seed": seed,
            "cfg": cfg,
            # Give the remote a little slack so it reports the timeout rather
            # than the socket dying underneath us.
            "timeout_s": max(timeout_s - 15.0, 30.0),
            **geometry_payload(aspect_ratio, megapixels, width, height),
        }

        # Ideogram 4 accepts a structured caption; if the prompt box already
        # holds one, forward it as JSON so magic-prompt stays disabled.
        stripped = prompt.strip()
        if stripped.startswith("{"):
            try:
                payload["json_prompt"] = json.loads(stripped)
            except json.JSONDecodeError:
                payload["prompt"] = prompt
        else:
            payload["prompt"] = prompt

        with ProgressMirror(url, client_id, unique_id):
            result = post(url, "/generate", payload, timeout_s)

        params = result.get("params", {})
        info = (
            f"seed={params.get('seed')} steps={params.get('steps')} "
            f"{params.get('width')}x{params.get('height')} in {result.get('duration_s')}s"
        )
        return (to_tensor(result["images"]), int(params.get("seed", seed)), info)


class Ideogram4ModalCaptionTemplate:
    """Fetch the magic-prompt template so an LLM can write the JSON caption.

    Ideogram 4 renders best from a structured caption. The deployment serves the
    same template ComfyUI's official workflow ships; paste the output into any
    instruction-following model and feed its JSON back into the prompt box.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
            },
            "optional": {"endpoint": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("llm_prompt",)
    FUNCTION = "fetch"
    CATEGORY = CATEGORY

    def fetch(self, prompt: str, width: int, height: int, endpoint: str = ""):
        url = _endpoint_url(endpoint)
        text = get(url, "/caption-template", {"prompt": prompt, "width": width, "height": height})
        return (text,)


def _endpoint_url(override: str) -> str:
    """The widget arg shadows the imported `endpoint`, hence the indirection."""
    return endpoint(override, ENV_URL)


NODE_CLASS_MAPPINGS = {
    "Ideogram4Modal": Ideogram4Modal,
    "Ideogram4ModalCaptionTemplate": Ideogram4ModalCaptionTemplate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ideogram4Modal": "Ideogram 4 (Modal)",
    "Ideogram4ModalCaptionTemplate": "Ideogram 4 Caption Template (Modal)",
}
