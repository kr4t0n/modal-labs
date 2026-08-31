"""WAI-illustrious-SDXL's slice of the ASGI layer.

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
async def test_generate_submits_an_sdxl_graph(wired):
    _, client, state = wired
    response = await client.post("/generate", json={"prompt": "1girl, solo", "seed": 5})
    assert response.status_code == 200, response.text
    assert base64.b64decode(response.json()["images"][0]["b64"]) == PNG

    submitted = state["prompts"][0]["prompt"]
    assert submitted["load_checkpoint"]["inputs"]["ckpt_name"] == workflow.CHECKPOINT
    assert submitted["positive"]["inputs"]["text"] == "1girl, solo"
    assert submitted["sample"]["inputs"]["seed"] == 5


@pytest.mark.asyncio
async def test_default_negative_and_clip_skip_are_applied(wired):
    """Omitting them must not mean 'no negative' or 'no clip skip'."""
    _, client, state = wired
    await client.post("/generate", json={"prompt": "1girl"})
    submitted = state["prompts"][0]["prompt"]
    assert submitted["negative"]["inputs"]["text"] == workflow.DEFAULT_NEGATIVE_PROMPT
    assert submitted["clip_skip"]["inputs"]["stop_at_clip_layer"] == -2


@pytest.mark.asyncio
async def test_negative_prompt_can_be_cleared(wired):
    _, client, state = wired
    await client.post("/generate", json={"prompt": "1girl", "negative_prompt": ""})
    assert state["prompts"][0]["prompt"]["negative"]["inputs"]["text"] == ""


@pytest.mark.asyncio
async def test_missing_prompt_is_rejected(wired):
    _, client, _ = wired
    assert (await client.post("/generate", json={"width": 512})).status_code == 422


@pytest.mark.asyncio
async def test_out_of_range_clip_skip_is_rejected_by_the_schema(wired):
    _, client, _ = wired
    assert (await client.post("/generate", json={"prompt": "x", "clip_skip": 0})).status_code == 422


@pytest.mark.asyncio
async def test_defaults_endpoint_reports_the_conventions(wired):
    _, client, _ = wired
    body = (await client.get("/defaults")).json()
    assert body["checkpoint"] == workflow.CHECKPOINT
    assert body["clip_skip"] == -2
    assert body["sampler_name"] == "euler_ancestral"
