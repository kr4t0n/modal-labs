"""A stub ComfyUI, so services can be tested offline without a GPU.

Every suite that exercises the ASGI layer needs the same fake upstream: one that
accepts a prompt, reports it as queued once, then completes with an image. It
lives in the package rather than in a test file so all three suites drive the
same double, and so the shape of what ComfyUI actually returns is written down
in one place.
"""

from __future__ import annotations

import base64
import gzip
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response

# A 1x1 PNG, small enough to inline and still a real image.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def make_stub_comfyui(output_node_id: str) -> tuple[FastAPI, dict[str, Any]]:
    """A fake ComfyUI plus a dict recording what it was asked to do.

    `output_node_id` must match the service's, since /history keys its outputs
    by node id and the real fetch path looks the image up that way.
    """
    upstream = FastAPI()
    state: dict[str, Any] = {"prompts": [], "polls": 0}

    @upstream.get("/system_stats")
    async def system_stats():
        return {"system": {"comfyui_version": "stub"}}

    @upstream.post("/prompt")
    async def prompt(request: Request):
        state["prompts"].append(await request.json())
        return {"prompt_id": "abc123", "number": 1}

    @upstream.get("/history/{prompt_id}")
    async def history(prompt_id: str):
        state["polls"] += 1
        # Answer "still running" once, so the polling path is exercised.
        if state["polls"] < 2:
            return {}
        return {
            prompt_id: {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    output_node_id: {
                        "images": [
                            {"filename": "out_00001_.png", "subfolder": "", "type": "output"}
                        ]
                    }
                },
            }
        }

    @upstream.get("/view")
    async def view(filename: str, subfolder: str = "", type: str = "output"):
        state["viewed"] = filename
        return Response(PNG, media_type="image/png")

    @upstream.get("/object_info")
    async def object_info():
        return {"UNETLoader": {"input": {"required": {"unet_name": [[]]}}}}

    @upstream.post("/upload/image")
    async def upload(request: Request):
        state["uploaded"] = await request.body()
        return {"name": "uploaded.png"}

    @upstream.get("/compressed")
    async def compressed():
        return Response(
            gzip.compress(b"ComfyUI frontend bundle"),
            media_type="text/plain",
            headers={"Content-Encoding": "gzip"},
        )

    return upstream, state


def wire(service_app: FastAPI, upstream: FastAPI) -> httpx.AsyncClient:
    """Point a service app at a stub upstream and return a client for it.

    Stands in for the ASGI lifespan, which would otherwise open a real socket.
    """
    service_app.state.http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url="http://comfy"
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=service_app), base_url="http://test")
