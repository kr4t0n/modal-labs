"""FLUX.2 klein's slice of the ASGI layer.

The generic surface — the proxy, the submit/poll/fetch loop, client_id
forwarding — is covered once in `tests/test_server.py` against a stub model.
This file asserts only what is specific to FLUX.2 klein.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Both services ship top-level modules with these names, and pytest collects
# every suite in one interpreter. Drop whatever the other project's suite left
# in sys.modules so the imports below resolve against *this* project.
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
async def test_variant_drives_checkpoint_and_schedule(wired):
    _, client, state = wired
    response = await client.post(
        "/generate", json={"prompt": "a test", "variant": "distilled", "seed": 5}
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["params"]["variant"] == "distilled"
    assert body["params"]["steps"] == 4
    assert body["params"]["cfg"] == 1.0
    assert base64.b64decode(body["images"][0]["b64"]) == PNG

    submitted = state["prompts"][0]["prompt"]
    assert submitted["sigmas"]["inputs"]["steps"] == 4
    assert submitted["load_unet"]["inputs"]["unet_name"] == "flux-2-klein-9b-fp8.safetensors"


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
async def test_missing_prompt_is_rejected(wired):
    _, client, _ = wired
    response = await client.post("/generate", json={"width": 512})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_explicit_steps_and_cfg_override_the_variant(wired):
    _, client, state = wired
    await client.post(
        "/generate", json={"prompt": "a test", "variant": "distilled", "steps": 9, "cfg": 3.5}
    )
    submitted = state["prompts"][0]["prompt"]
    assert submitted["sigmas"]["inputs"]["steps"] == 9
    assert submitted["guider"]["inputs"]["cfg"] == 3.5


@pytest.mark.asyncio
async def test_uncensored_variant_swaps_only_the_encoder(wired):
    _, client, state = wired
    await client.post("/generate", json={"prompt": "a test", "variant": "ponpoke-uncensored"})
    submitted = state["prompts"][0]["prompt"]
    assert submitted["load_clip"]["inputs"]["clip_name"] == workflow.UNCENSORED_TEXT_ENCODER
    assert submitted["load_unet"]["inputs"]["unet_name"] == "flux-2-klein-base-9b-fp8.safetensors"


@pytest.mark.asyncio
async def test_variants_endpoint_reports_sampler_defaults(wired):
    _, client, _ = wired
    body = (await client.get("/variants")).json()
    assert body["variants"]["base"]["steps"] == 20
    assert body["variants"]["distilled"]["steps"] == 4
    assert body["default"] == "base"
