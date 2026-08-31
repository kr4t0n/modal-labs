"""End-to-end tests for the ASGI layer against a stubbed ComfyUI.

These run offline: the "ComfyUI" upstream is a small ASGI app wired to the real
server through an in-memory transport. They cover the two things that only fail
in production otherwise — the submit/poll/fetch sequence in `/generate`, and
whether the catch-all really is a transparent proxy.
"""

from __future__ import annotations

import base64
import gzip
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request, Response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Both services ship top-level modules with these names, and pytest collects
# every suite in one interpreter. Drop whatever the other project's suite left
# in sys.modules so the imports below resolve against *this* project.
for _shared in ("workflow", "server", "app"):
    sys.modules.pop(_shared, None)

import server  # noqa: E402
import workflow  # noqa: E402

# A 1x1 PNG, small enough to inline and still a real image.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def make_upstream() -> tuple[FastAPI, dict]:
    """A stub ComfyUI. `state` records what it was asked to do."""
    upstream = FastAPI()
    state: dict = {"prompts": [], "polls": 0}

    @upstream.get("/system_stats")
    async def system_stats():
        return {"system": {"comfyui_version": "stub"}}

    @upstream.post("/prompt")
    async def prompt(request: Request):
        body = await request.json()
        state["prompts"].append(body)
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
                    workflow.OUTPUT_NODE_ID: {
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


@pytest.fixture
def wired():
    upstream, state = make_upstream()
    app = server.create_app("http://comfy")
    # Stand in for the lifespan, pointing the app at the stub over an
    # in-memory transport instead of a socket.
    app.state.http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=upstream), base_url="http://comfy"
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    return app, client, state


@pytest.mark.asyncio
async def test_generate_submits_polls_and_returns_the_image(wired):
    _, client, state = wired
    response = await client.post(
        "/generate", json={"prompt": "a test", "variant": "distilled", "seed": 5}
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["prompt_id"] == "abc123"
    assert body["params"]["seed"] == 5
    assert body["params"]["steps"] == 4
    assert body["params"]["cfg"] == 1.0
    assert body["params"]["variant"] == "distilled"
    assert base64.b64decode(body["images"][0]["b64"]) == PNG

    # The graph really was sent, and polling waited for completion.
    submitted = state["prompts"][0]["prompt"]
    assert submitted["sigmas"]["inputs"]["steps"] == 4
    assert submitted["positive"]["inputs"]["text"] == "a test"
    assert state["polls"] >= 2
    assert state["viewed"] == "out_00001_.png"


@pytest.mark.asyncio
async def test_client_id_is_forwarded_to_comfyui(wired):
    """The caller's id must reach ComfyUI, or progress events go nowhere."""
    _, client, state = wired
    await client.post("/generate", json={"prompt": "a test", "client_id": "node-42"})
    assert state["prompts"][0]["client_id"] == "node-42"


@pytest.mark.asyncio
async def test_client_id_is_generated_when_omitted(wired):
    _, client, state = wired
    await client.post("/generate", json={"prompt": "a test"})
    assert state["prompts"][0]["client_id"]


@pytest.mark.asyncio
async def test_generate_image_returns_raw_bytes(wired):
    _, client, _ = wired
    response = await client.post("/generate/image", json={"prompt": "a test"})
    assert response.status_code == 200
    assert response.content == PNG
    assert response.headers["content-type"] == "image/png"
    assert int(response.headers["X-Ideogram4-Seed"]) >= 0


@pytest.mark.asyncio
async def test_missing_prompt_is_rejected(wired):
    _, client, _ = wired
    response = await client.post("/generate", json={"width": 512})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_negative_prompt_reaches_its_own_encoder(wired):
    _, client, state = wired
    await client.post(
        "/generate", json={"prompt": "a cat", "negative_prompt": "blurry", "variant": "base"}
    )
    submitted = state["prompts"][0]["prompt"]
    assert submitted["positive"]["inputs"]["text"] == "a cat"
    assert submitted["negative"]["inputs"]["text"] == "blurry"


@pytest.mark.asyncio
async def test_health_reports_upstream(wired):
    _, client, _ = wired
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["system_stats"]["system"]["comfyui_version"] == "stub"


@pytest.mark.asyncio
async def test_unknown_paths_proxy_through(wired):
    _, client, _ = wired
    response = await client.get("/object_info")
    assert response.status_code == 200
    assert "UNETLoader" in response.json()


@pytest.mark.asyncio
async def test_proxy_forwards_request_bodies(wired):
    _, client, state = wired
    response = await client.post("/upload/image", content=b"raw-bytes")
    assert response.status_code == 200
    assert state["uploaded"] == b"raw-bytes"


@pytest.mark.asyncio
async def test_proxy_preserves_content_encoding(wired):
    """Raw pass-through: a gzipped body must keep its Content-Encoding header."""
    _, client, _ = wired
    response = await client.get("/compressed")
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert response.content == b"ComfyUI frontend bundle"  # httpx decodes it


@pytest.mark.asyncio
async def test_workflow_endpoint_returns_the_graph(wired):
    _, client, _ = wired
    response = await client.post("/workflow", json={"prompt": "a test", "aspect_ratio": "16:9"})
    body = response.json()
    assert body["params"]["width"] > body["params"]["height"]
    assert body["workflow"]["load_unet"]["class_type"] == "UNETLoader"


@pytest.mark.asyncio
async def test_variants_endpoint_reports_sampler_defaults(wired):
    _, client, _ = wired
    body = (await client.get("/variants")).json()
    assert body["variants"]["base"]["steps"] == 20
    assert body["variants"]["distilled"]["steps"] == 4
