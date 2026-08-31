"""End-to-end tests for the shared ASGI layer against a stubbed ComfyUI.

These run offline: the "ComfyUI" upstream is a small ASGI app wired to the real
server through an in-memory transport. They cover the two things that only fail
in production otherwise — the submit/poll/fetch sequence behind `/generate`, and
whether the catch-all really is a transparent proxy.

The model under test is a stub. Anything model-specific belongs in the per-service
suites; this file must stay ignorant of which model is being served.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import pytest

from comfyui_modal import geometry
from comfyui_modal.server import BaseGenerateRequest, ModelService, create_app
from comfyui_modal.testing import PNG, make_stub_comfyui, wire

OUTPUT_NODE = "save_image"


class StubRequest(BaseGenerateRequest):
    prompt: str = "a stub prompt"


@dataclass
class StubParams:
    prompt: str
    seed: int
    width: int
    height: int
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            **self.extras,
        }


def _resolve(request: StubRequest) -> StubParams:
    width, height = request.dimensions()
    return StubParams(request.prompt, geometry.normalise_seed(request.seed), width, height)


def _build(params: StubParams) -> dict[str, Any]:
    return {
        OUTPUT_NODE: {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "stub", "seed": params.seed},
        }
    }


STUB_SERVICE = ModelService(
    title="stub",
    description="stub service for testing the shared layer",
    request_model=StubRequest,
    resolve=_resolve,
    build_workflow=_build,
    output_node_id=OUTPUT_NODE,
)


@pytest.fixture
def wired():
    upstream, state = make_stub_comfyui(OUTPUT_NODE)
    app = create_app("http://comfy", STUB_SERVICE)
    return app, wire(app, upstream), state


@pytest.mark.asyncio
async def test_generate_submits_polls_and_returns_the_image(wired):
    _, client, state = wired
    response = await client.post("/generate", json={"prompt": "a test", "seed": 5})
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["prompt_id"] == "abc123"
    assert body["params"]["seed"] == 5
    assert base64.b64decode(body["images"][0]["b64"]) == PNG

    # The graph really was sent, and polling waited for completion.
    assert state["prompts"][0]["prompt"][OUTPUT_NODE]["inputs"]["seed"] == 5
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
    assert int(response.headers["X-Seed"]) >= 0
    assert response.headers["X-Prompt-Id"] == "abc123"


@pytest.mark.asyncio
async def test_health_reports_upstream(wired):
    _, client, _ = wired
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["system_stats"]["system"]["comfyui_version"] == "stub"


@pytest.mark.asyncio
async def test_workflow_endpoint_returns_the_graph_without_running_it(wired):
    _, client, state = wired
    response = await client.post("/workflow", json={"prompt": "a test", "aspect_ratio": "21:9"})
    body = response.json()
    assert body["params"]["width"] > body["params"]["height"]
    assert OUTPUT_NODE in body["workflow"]
    assert state["prompts"] == []


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
async def test_resolution_endpoint_lists_the_shared_geometry(wired):
    _, client, _ = wired
    body = (await client.get("/resolution")).json()
    assert body["multiple_of"] == geometry.SIDE_MULTIPLE
    assert "16:9" in body["aspect_ratios"]
