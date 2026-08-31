"""Z-Image Turbo's slice of the ASGI layer.

The generic surface is covered once in `tests/test_server.py` against a stub
model; this file asserts only what is specific to this service.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _shared in ("workflow", "server", "app"):
    sys.modules.pop(_shared, None)

import server  # noqa: E402
import workflow  # noqa: E402
from comfyui_modal.testing import PNG, make_stub_comfyui, wire  # noqa: E402


@pytest.fixture
def wired():
    upstream, state = make_stub_comfyui(workflow.OUTPUT_NODE_ID)
    app = server.create_app("http://comfy")
    return app, wire(app, upstream), state


@pytest.mark.asyncio
async def test_generate_submits_the_turbo_defaults(wired):
    _, client, state = wired
    response = await client.post("/generate", json={"prompt": "a test", "seed": 5})
    assert response.status_code == 200, response.text
    assert base64.b64decode(response.json()["images"][0]["b64"]) == PNG

    submitted = state["prompts"][0]["prompt"]
    assert submitted["load_unet"]["inputs"]["unet_name"] == workflow.DIFFUSION_MODEL
    assert submitted["sample"]["inputs"]["steps"] == 8
    assert submitted["sample"]["inputs"]["cfg"] == 1.0
    assert submitted["sample"]["inputs"]["sampler_name"] == "res_multistep"
    # The sampling-shift patch must survive the round trip.
    assert submitted["model_sampling"]["inputs"]["shift"] == 3.0
    assert submitted["positive"]["inputs"]["text"] == "a test"


@pytest.mark.asyncio
async def test_omitting_a_negative_zeroes_the_conditioning(wired):
    _, client, state = wired
    await client.post("/generate", json={"prompt": "a test"})
    node = state["prompts"][0]["prompt"][workflow.NEGATIVE_NODE_ID]
    assert node["class_type"] == "ConditioningZeroOut"


@pytest.mark.asyncio
async def test_supplying_a_negative_encodes_it(wired):
    _, client, state = wired
    await client.post("/generate", json={"prompt": "a test", "negative_prompt": "blurry", "cfg": 3})
    node = state["prompts"][0]["prompt"][workflow.NEGATIVE_NODE_ID]
    assert node["class_type"] == "CLIPTextEncode"
    assert node["inputs"]["text"] == "blurry"


@pytest.mark.asyncio
async def test_missing_prompt_is_rejected(wired):
    _, client, _ = wired
    assert (await client.post("/generate", json={"width": 512})).status_code == 422


@pytest.mark.asyncio
async def test_defaults_endpoint_names_its_source(wired):
    _, client, _ = wired
    body = (await client.get("/defaults")).json()
    assert body["steps"] == 8 and body["cfg"] == 1.0
    assert body["diffusion_model"] == workflow.DIFFUSION_MODEL
    assert body["shift"] == 3.0
    assert "template" in body["source"]
