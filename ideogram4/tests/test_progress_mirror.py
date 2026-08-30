"""Tests for the custom node's remote progress mirroring.

The node normally runs inside ComfyUI, so `torch`, `numpy` and `comfy.utils`
are stubbed here — the progress path touches none of them. What is *not*
stubbed is the websocket: a real aiohttp server emits ComfyUI-shaped progress
events and the mirror is asserted to turn them into progress-bar updates.

Worth testing precisely because `_ProgressMirror` swallows all of its own
exceptions by design: if it silently stopped working, nothing would say so.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import ClassVar

import pytest
from aiohttp import web

NODE_SOURCE = Path(__file__).resolve().parents[1] / "comfy_node" / "nodes.py"


class RecordingProgressBar:
    """Stands in for comfy.utils.ProgressBar, recording what the UI would draw."""

    instances: ClassVar[list[RecordingProgressBar]] = []

    def __init__(self, total, node_id=None):
        self.total = total
        self.node_id = node_id
        self.updates: list[tuple[int, int]] = []
        RecordingProgressBar.instances.append(self)

    def update_absolute(self, value, total=None, preview=None):
        self.updates.append((value, total if total is not None else self.total))


def load_node_module():
    """Import comfy_node/nodes.py with ComfyUI's environment faked out."""
    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = RecordingProgressBar
    comfy.utils = comfy_utils

    torch = types.ModuleType("torch")
    numpy = types.ModuleType("numpy")

    sys.modules.update({"comfy": comfy, "comfy.utils": comfy_utils, "torch": torch, "numpy": numpy})
    spec = importlib.util.spec_from_file_location("ideogram4_node_under_test", NODE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nodes = load_node_module()


@pytest.fixture(autouse=True)
def clear_recorded_bars():
    RecordingProgressBar.instances.clear()


async def start_ws_server(events, seen):
    """A stub ComfyUI websocket that replays `events` to whoever connects."""

    async def handler(request):
        seen["client_id"] = request.rel_url.query.get("clientId")
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        for event in events:
            await socket.send_json(event)
        seen["sent"] = True
        # Drain until the mirror closes the socket. Without consuming messages
        # the close frame is never processed and shutdown blocks.
        async for _ in socket:
            pass
        return socket

    app = web.Application()
    app.router.add_get("/ws", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = next(iter(runner.addresses))[1]
    return runner, f"http://127.0.0.1:{port}"


def wait_for(predicate, timeout=5.0):
    """Block until `predicate` holds. Synchronous: it runs off the event loop."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def progress(value, maximum):
    return {"type": "progress", "data": {"value": value, "max": maximum, "node": "7"}}


@pytest.mark.asyncio
async def test_progress_events_drive_the_local_bar():
    seen: dict = {}
    events = [progress(1, 20), progress(10, 20), progress(20, 20)]
    runner, url = await start_ws_server(events, seen)

    def drive():
        # Off the event loop: the mirror's context manager blocks, and the stub
        # server shares this test's loop.
        started = time.monotonic()
        with nodes._ProgressMirror(url, "client-abc", "7"):
            mirrored = wait_for(
                lambda: (
                    RecordingProgressBar.instances
                    and len(RecordingProgressBar.instances[0].updates) == 3
                )
            )
        return mirrored, time.monotonic() - started

    try:
        mirrored, elapsed = await asyncio.to_thread(drive)
    finally:
        await runner.cleanup()

    assert mirrored, "progress events were not mirrored"
    # Connect and teardown must both be prompt: the node holds the render open
    # across __exit__, so a slow join would tax every generation.
    assert elapsed < 1.0, f"mirror setup/teardown took {elapsed:.2f}s"

    bar = RecordingProgressBar.instances[0]
    assert bar.updates == [(1, 20), (10, 20), (20, 20)]
    assert bar.total == 20
    # The bar must be bound to the node so the UI draws it in the right place.
    assert bar.node_id == "7"
    # And it must have subscribed with the id it will hand to /generate.
    assert seen["client_id"] == "client-abc"


@pytest.mark.asyncio
async def test_non_progress_traffic_is_ignored():
    seen: dict = {}
    events = [
        {"type": "status", "data": {"status": {}}},
        {"type": "executing", "data": {"node": "7"}},
        progress(4, 12),
        {"type": "progress", "data": {"value": 5}},  # malformed: no max
        {"type": "progress", "data": {"value": 6, "max": 0}},  # nonsense total
    ]
    runner, url = await start_ws_server(events, seen)

    def drive():
        with nodes._ProgressMirror(url, "c", None):
            assert wait_for(lambda: seen.get("sent"))
            time.sleep(0.2)

    try:
        await asyncio.to_thread(drive)
    finally:
        await runner.cleanup()

    assert len(RecordingProgressBar.instances) == 1
    assert RecordingProgressBar.instances[0].updates == [(4, 12)]


def test_unreachable_endpoint_neither_raises_nor_stalls():
    """Progress is cosmetic: a dead socket must not delay or break a render."""
    started = time.monotonic()
    with nodes._ProgressMirror("http://127.0.0.1:1", "c", "7"):
        pass
    elapsed = time.monotonic() - started

    assert elapsed < nodes._ProgressMirror.CONNECT_TIMEOUT_S
    assert RecordingProgressBar.instances == []


def test_generate_sends_a_client_id_matching_the_subscription(monkeypatch):
    """The id posted to /generate must be the one the mirror subscribes with."""
    captured: dict = {}

    def fake_post(url, path, payload, timeout):
        captured["payload"] = payload
        return {"images": [], "params": {"seed": 1}, "duration_s": 0.1}

    subscriptions: list[str] = []

    class SpyMirror(nodes._ProgressMirror):
        def __init__(self, url, client_id, node_id):
            subscriptions.append(client_id)
            super().__init__(url, client_id, node_id)

    monkeypatch.setattr(nodes, "_post", fake_post)
    monkeypatch.setattr(nodes, "_ProgressMirror", SpyMirror)
    monkeypatch.setattr(nodes, "_to_tensor", lambda images: images)

    nodes.Ideogram4Modal().generate(
        prompt="a test",
        preset="Turbo",
        aspect_ratio="1:1",
        megapixels=1.0,
        width=1024,
        height=1024,
        batch_size=1,
        seed=3,
        cfg=7.0,
        endpoint="http://127.0.0.1:1",
        unique_id="9",
    )

    assert subscriptions == [captured["payload"]["client_id"]]


def test_progress_bar_without_node_id_support():
    """Older ComfyUI builds have no `node_id` kwarg; the bar must still draw."""
    created: list[int] = []

    class LegacyProgressBar:
        def __init__(self, total):
            created.append(total)
            self.total = total
            self.updates: list[tuple[int, int]] = []

        def update_absolute(self, value, total=None, preview=None):
            self.updates.append((value, total))

    mirror = nodes._ProgressMirror("http://127.0.0.1:1", "c", "7")
    original = nodes.comfy.utils.ProgressBar
    nodes.comfy.utils.ProgressBar = LegacyProgressBar
    try:
        bar = mirror._apply(progress(3, 8), None)
    finally:
        nodes.comfy.utils.ProgressBar = original

    assert created == [8]
    assert bar.updates == [(3, 8)]
