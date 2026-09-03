"""FinePorn's slice of the ASGI layer.

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
async def test_generate_submits_the_model_cards_defaults(wired):
    _, client, state = wired
    response = await client.post("/generate", json={"prompt": "a test", "seed": 5})
    assert response.status_code == 200, response.text
    assert base64.b64decode(response.json()["images"][0]["b64"]) == PNG

    submitted = state["prompts"][0]["prompt"]
    assert submitted["load_unet"]["inputs"]["unet_name"] == workflow.DIFFUSION_MODEL
    assert submitted["sample"]["inputs"]["steps"] == 10
    assert submitted["sample"]["inputs"]["cfg"] == 1.0
    assert submitted["sample"]["inputs"]["sampler_name"] == "euler"
    assert submitted["sample"]["inputs"]["scheduler"] == "beta"
    assert submitted["positive"]["inputs"]["text"] == "a test"


@pytest.mark.asyncio
async def test_a_bare_request_renders_above_one_megapixel(wired):
    """A caller who names no size must still get the merge's native resolution."""
    _, client, state = wired
    await client.post("/generate", json={"prompt": "a test"})
    latent = state["prompts"][0]["prompt"]["latent"]["inputs"]
    assert (latent["width"], latent["height"]) == (1280, 1280)


@pytest.mark.asyncio
async def test_explicit_dimensions_still_win(wired):
    _, client, state = wired
    await client.post("/generate", json={"prompt": "a test", "width": 832, "height": 1216})
    latent = state["prompts"][0]["prompt"]["latent"]["inputs"]
    assert (latent["width"], latent["height"]) == (832, 1216)


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
async def test_defaults_endpoint_names_its_source_and_the_resolutions(wired):
    _, client, _ = wired
    body = (await client.get("/defaults")).json()
    assert body["steps"] == 10 and body["cfg"] == 1.0
    assert body["sampler_name"] == "euler" and body["scheduler"] == "beta"
    assert body["diffusion_model"] == workflow.DIFFUSION_MODEL
    # Attributed to the card rather than a ComfyUI template, unlike ultra.
    assert "model card" in body["source"]
    assert body["width"] == body["height"] == 1280
    assert {"standard": [1024, 1024], "optimal": [1280, 1280], "recommended": [1536, 1536]} in body[
        "recommended_resolutions"
    ]
    # Reported so a caller can follow it; never injected into the prompt.
    assert "casual" in body["prompt_guidance"]


@pytest.mark.asyncio
async def test_prompt_guidance_is_advice_not_an_injection(wired):
    """The card's opener is surfaced, but the graph sends exactly what was asked."""
    _, client, state = wired
    await client.post("/generate", json={"prompt": "a portrait"})
    assert state["prompts"][0]["prompt"]["positive"]["inputs"]["text"] == "a portrait"
