"""Ideogram 4's slice of the ASGI layer.

The generic surface — the proxy, the submit/poll/fetch loop, client_id
forwarding — is covered once in `tests/test_server.py` against a stub model.
This file asserts only what is specific to Ideogram 4.
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

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "magic_prompt_template.txt"


@pytest.fixture
def wired():
    upstream, state = make_stub_comfyui(workflow.OUTPUT_NODE_ID)
    app = server.create_app("http://comfy", ASSETS)
    return app, wire(app, upstream), state


@pytest.mark.asyncio
async def test_preset_drives_the_submitted_schedule(wired):
    _, client, state = wired
    response = await client.post(
        "/generate", json={"prompt": "a test", "preset": "Turbo", "seed": 5}
    )
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["params"]["seed"] == 5
    assert body["params"]["steps"] == 12
    assert base64.b64decode(body["images"][0]["b64"]) == PNG

    submitted = state["prompts"][0]["prompt"]
    assert submitted["sigmas"]["inputs"]["steps"] == 12
    assert submitted["positive"]["inputs"]["text"] == "a test"


@pytest.mark.asyncio
async def test_structured_caption_is_serialised_into_the_encoder(wired):
    _, client, state = wired
    await client.post("/generate", json={"json_prompt": {"high_level_description": "a bee"}})
    encoded = state["prompts"][0]["prompt"]["positive"]["inputs"]["text"]
    assert "high_level_description" in encoded and "a bee" in encoded


@pytest.mark.asyncio
async def test_neither_prompt_nor_caption_is_rejected(wired):
    _, client, _ = wired
    response = await client.post("/generate", json={"width": 512})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_caption_template_is_filled_in(wired):
    _, client, _ = wired
    response = await client.get(
        "/caption-template", params={"prompt": "a bee", "width": 1000, "height": 500}
    )
    assert response.status_code == 200
    assert "{{original_prompt}}" not in response.text
    assert "a bee" in response.text


@pytest.mark.asyncio
async def test_presets_endpoint_reports_the_schedule_table(wired):
    _, client, _ = wired
    body = (await client.get("/presets")).json()
    assert body["sampling_presets"]["Turbo"]["steps"] == 12
    assert body["sampling_presets"]["Quality"]["steps"] == 48
