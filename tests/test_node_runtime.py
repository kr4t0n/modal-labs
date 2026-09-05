"""Tests for the ComfyUI custom-node package.

The package normally runs inside the user's ComfyUI, so `torch`, `numpy` and
`comfy.utils` are stubbed here — the paths under test touch none of them. What
is *not* stubbed is the websocket: a real aiohttp server emits ComfyUI-shaped
progress events and the mirror is asserted to turn them into bar updates.

Worth testing precisely because `ProgressMirror` swallows all of its own
exceptions by design: if it silently stopped working, nothing would say so.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import sys
import time
from pathlib import Path
from typing import ClassVar

import pytest
from aiohttp import web

from comfyui_modal.testing import install_comfyui_stubs


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


# Assigns rather than defaults, so this suite gets its recorder regardless of
# whether another test module stubbed `comfy` first. See install_comfyui_stubs.
install_comfyui_stubs(RecordingProgressBar)

import comfy_node  # noqa: E402
from comfy_node import _runtime  # noqa: E402


@pytest.fixture(autouse=True)
def clear_recorded_bars():
    RecordingProgressBar.instances.clear()


# --- The node contract ------------------------------------------------------
# Node ids and widget names are what a saved workflow JSON references. Renaming
# or reordering one silently breaks every workflow a user has saved, so they are
# pinned here rather than left to drift.


def test_every_service_registers_its_node():
    assert set(comfy_node.NODE_CLASS_MAPPINGS) == {
        "Flux2KleinModal",
        "UltraModal",
        "ZImageTurboStableYogiModal",
        "FinePornV4Modal",
        "RedGPT2GPTModal",
        "RedCraft3Modal",
        "DarkBeast3Modal",
    }
    assert set(comfy_node.NODE_DISPLAY_NAME_MAPPINGS) == set(comfy_node.NODE_CLASS_MAPPINGS)


@pytest.mark.parametrize(
    ("node_id", "expected"),
    [
        (
            "Flux2KleinModal",
            [
                "prompt",
                "negative_prompt",
                "variant",
                "aspect_ratio",
                "megapixels",
                "width",
                "height",
                "batch_size",
                "seed",
                "override_sampler",
                "steps",
                "cfg",
                "lora",
                "lora_strength",
                "override_lora_strength",
            ],
        ),
        (
            "UltraModal",
            [
                "prompt",
                "negative_prompt",
                "aspect_ratio",
                "megapixels",
                "width",
                "height",
                "batch_size",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
            ],
        ),
        (
            "FinePornV4Modal",
            [
                "prompt",
                "negative_prompt",
                "aspect_ratio",
                "megapixels",
                "width",
                "height",
                "batch_size",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
            ],
        ),
        (
            "RedGPT2GPTModal",
            [
                "prompt",
                "negative_prompt",
                "aspect_ratio",
                "megapixels",
                "width",
                "height",
                "batch_size",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
            ],
        ),
        (
            "RedCraft3Modal",
            [
                "prompt",
                "negative_prompt",
                "aspect_ratio",
                "megapixels",
                "width",
                "height",
                "batch_size",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
            ],
        ),
        (
            "DarkBeast3Modal",
            [
                "prompt",
                "negative_prompt",
                "aspect_ratio",
                "megapixels",
                "width",
                "height",
                "batch_size",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
            ],
        ),
        (
            "ZImageTurboStableYogiModal",
            [
                "prompt",
                "negative_prompt",
                "aspect_ratio",
                "megapixels",
                "width",
                "height",
                "batch_size",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "shift",
            ],
        ),
    ],
)
def test_widget_names_and_order_are_stable(node_id, expected):
    schema = comfy_node.NODE_CLASS_MAPPINGS[node_id].INPUT_TYPES()
    assert list(schema["required"]) == expected
    assert schema["hidden"] == {"unique_id": "UNIQUE_ID"}
    # Transport is on every node; a node may add its own optional inputs beyond
    # it, which is how klein takes a reference image without disturbing others.
    assert {"endpoint", "timeout_s"} <= set(schema["optional"])


@pytest.mark.parametrize(
    ("node_id", "extra"),
    [
        ("Flux2KleinModal", {"reference_image"}),
        ("UltraModal", set()),
        ("ZImageTurboStableYogiModal", set()),
        ("FinePornV4Modal", set()),
        ("RedGPT2GPTModal", set()),
        ("RedCraft3Modal", set()),
        ("DarkBeast3Modal", set()),
    ],
)
def test_optional_inputs_beyond_transport_are_pinned(node_id, extra):
    """An accidental optional input is a silently changed node contract."""
    schema = comfy_node.NODE_CLASS_MAPPINGS[node_id].INPUT_TYPES()
    assert set(schema["optional"]) - {"endpoint", "timeout_s"} == extra


def test_the_klein_reference_input_is_an_optional_image():
    """Optional and IMAGE-typed: a saved text-to-image workflow must still load."""
    schema = comfy_node.NODE_CLASS_MAPPINGS["Flux2KleinModal"].INPUT_TYPES()
    assert schema["optional"]["reference_image"][0] == "IMAGE"
    assert "reference_image" not in schema["required"]


def test_nodes_return_an_image_seed_and_info():
    for node_id in (
        "Flux2KleinModal",
        "UltraModal",
        "ZImageTurboStableYogiModal",
        "FinePornV4Modal",
        "RedGPT2GPTModal",
        "RedCraft3Modal",
        "DarkBeast3Modal",
    ):
        node = comfy_node.NODE_CLASS_MAPPINGS[node_id]
        assert node.RETURN_TYPES == ("IMAGE", "INT", "STRING")
        assert node.RETURN_NAMES == ("image", "seed", "info")


# --- Settings ---------------------------------------------------------------


def test_endpoint_prefers_the_override_then_the_environment(monkeypatch):
    monkeypatch.setenv("DEMO_URL", "https://from-env.modal.run/")
    assert _runtime.endpoint("", "DEMO_URL") == "https://from-env.modal.run"
    assert _runtime.endpoint("https://override.run/", "DEMO_URL") == "https://override.run"


def test_missing_endpoint_names_the_variable(monkeypatch):
    monkeypatch.delenv("DEMO_URL", raising=False)
    monkeypatch.setattr(_runtime, "_dotenv", dict)
    with pytest.raises(RuntimeError, match="DEMO_URL"):
        _runtime.endpoint("", "DEMO_URL")


def test_auth_headers_omitted_unless_both_halves_present(monkeypatch):
    monkeypatch.setattr(_runtime, "_dotenv", dict)
    monkeypatch.delenv("MODAL_KEY", raising=False)
    monkeypatch.delenv("MODAL_SECRET", raising=False)
    assert _runtime.headers() == {}

    monkeypatch.setenv("MODAL_KEY", "wk-1")
    assert _runtime.headers() == {}, "half a credential must not be sent"

    monkeypatch.setenv("MODAL_SECRET", "ws-1")
    assert _runtime.headers() == {"Modal-Key": "wk-1", "Modal-Secret": "ws-1"}


def test_geometry_payload_drops_ratio_when_custom():
    assert _runtime.geometry_payload("custom", 2.0, 800, 600) == {"width": 800, "height": 600}
    payload = _runtime.geometry_payload("16:9", 2.0, 800, 600)
    assert payload["aspect_ratio"] == "16:9" and payload["megapixels"] == 2.0


# --- Progress mirroring -----------------------------------------------------


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
    runner, url = await start_ws_server([progress(1, 20), progress(10, 20), progress(20, 20)], seen)

    def drive():
        # Off the event loop: the mirror's context manager blocks, and the stub
        # server shares this test's loop.
        started = time.monotonic()
        with _runtime.ProgressMirror(url, "client-abc", "7"):
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
    assert bar.node_id == "7"
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
        with _runtime.ProgressMirror(url, "c", None):
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
    with _runtime.ProgressMirror("http://127.0.0.1:1", "c", "7"):
        pass
    elapsed = time.monotonic() - started

    assert elapsed < _runtime.ProgressMirror.CONNECT_TIMEOUT_S
    assert RecordingProgressBar.instances == []


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

    mirror = _runtime.ProgressMirror("http://127.0.0.1:1", "c", "7")
    original = _runtime.comfy.utils.ProgressBar
    _runtime.comfy.utils.ProgressBar = LegacyProgressBar
    try:
        bar = mirror._apply(progress(3, 8), None)
    finally:
        _runtime.comfy.utils.ProgressBar = original

    assert created == [8]
    assert bar.updates == [(3, 8)]


# --- Registry mirrors -------------------------------------------------------
# The node hardcodes its variant and adapter names. It has to: it runs inside
# the user's ComfyUI, where the deployment's `workflow` module does not exist,
# and ComfyUI calls INPUT_TYPES() during startup, so fetching /variants there
# would block the boot and fail whenever the endpoint is down. That makes the
# lists a hand-maintained mirror, and drift is silent until a render fails with
# `unknown lora`. These tests are the thing that notices.


@contextlib.contextmanager
def service_workflow(service: str):
    """Import one service's `workflow` module, then put `sys.modules` back.

    Every service ships a top-level `workflow`, `server` and `app`, and pytest
    collects them all in one interpreter, so borrowing the name has to be undone
    or the next suite silently tests this one's graph. See flux2klein/AGENTS.md.
    """
    directory = str(Path(__file__).resolve().parents[1] / service)
    saved = {name: sys.modules.pop(name, None) for name in ("workflow", "server", "app")}
    sys.path.insert(0, directory)
    try:
        yield importlib.import_module("workflow")
    finally:
        sys.path.remove(directory)
        sys.modules.pop("workflow", None)
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


def test_node_adapter_names_match_the_service_registry():
    offered = comfy_node.nodes_flux2klein.LORAS
    with service_workflow("flux2klein") as workflow:
        registered = sorted(workflow.LORAS)

    # "none" is the node's own affordance, not a registry entry: the widget
    # needs an explicit way to say "no adapter".
    assert offered[0] == "none"
    assert sorted(offered[1:]) == registered
    assert len(set(offered)) == len(offered)


def test_node_variant_names_match_the_service_registry():
    with service_workflow("flux2klein") as workflow:
        registered = sorted(workflow.VARIANTS)
    assert sorted(comfy_node.nodes_flux2klein.VARIANTS) == registered


def test_finepornv4_node_mirrors_the_services_resolution_defaults():
    """This service renders above 1 MP, and the widgets always send a value.

    A node left at the shared 1024/1.0 defaults would override the server on
    every call, quietly undoing the one thing this deployment tunes.
    """
    with service_workflow("finepornv4") as workflow:
        assert comfy_node.nodes_finepornv4.DEFAULT_SIDE == workflow.DEFAULT_SIDE
        assert comfy_node.nodes_finepornv4.DEFAULT_MEGAPIXELS == workflow.DEFAULT_MEGAPIXELS
        expected_sampler = workflow.DEFAULT_SAMPLER
        expected_scheduler = workflow.DEFAULT_SCHEDULER
        expected_steps = workflow.DEFAULT_STEPS

    widgets = comfy_node.NODE_CLASS_MAPPINGS["FinePornV4Modal"].INPUT_TYPES()["required"]
    assert widgets["width"][1]["default"] == comfy_node.nodes_finepornv4.DEFAULT_SIDE
    assert widgets["height"][1]["default"] == comfy_node.nodes_finepornv4.DEFAULT_SIDE
    assert widgets["megapixels"][1]["default"] == comfy_node.nodes_finepornv4.DEFAULT_MEGAPIXELS
    assert widgets["steps"][1]["default"] == expected_steps
    # The dropdowns must *offer* the recipe pairing and default to it.
    assert widgets["sampler_name"][1]["default"] == expected_sampler
    assert widgets["scheduler"][1]["default"] == expected_scheduler
    assert expected_sampler in widgets["sampler_name"][0]
    assert expected_scheduler in widgets["scheduler"][0]


def test_shared_geometry_defaults_are_unchanged_for_other_nodes():
    """The override is keyword-only; every other node keeps 1024/1.0."""
    for node_id in (
        "Flux2KleinModal",
        "UltraModal",
        "ZImageTurboStableYogiModal",
        "RedGPT2GPTModal",
        "RedCraft3Modal",
        "DarkBeast3Modal",
    ):
        widgets = comfy_node.NODE_CLASS_MAPPINGS[node_id].INPUT_TYPES()["required"]
        assert widgets["width"][1]["default"] == 1024
        assert widgets["height"][1]["default"] == 1024
        assert widgets["megapixels"][1]["default"] == 1.0


# --- Reference images -------------------------------------------------------
# `from_tensor` does real pixel work, but numpy/torch are container-only in this
# repo (see the root AGENTS.md on the thin local environment), so its encoding
# is not exercised here. What *is* pinned is the wiring around it: what the node
# puts in the payload, and when.


class FakeBatch(list):
    """Enough of an IMAGE batch for the node: length and slicing."""

    def __getitem__(self, item):
        result = list.__getitem__(self, item)
        return FakeBatch(result) if isinstance(item, slice) else result


def drive_klein(monkeypatch, **overrides):
    """Run the klein node against a stubbed transport and return the payload."""
    sent = {}
    node = comfy_node.NODE_CLASS_MAPPINGS["Flux2KleinModal"]()
    monkeypatch.setattr(
        comfy_node.nodes_flux2klein,
        "from_tensor",
        lambda batch: [f"b64-{i}" for i in range(len(batch))],
    )
    monkeypatch.setattr(comfy_node.nodes_flux2klein, "to_tensor", lambda images: "TENSOR")
    monkeypatch.setattr(
        comfy_node.nodes_flux2klein,
        "ProgressMirror",
        lambda *a, **k: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        comfy_node.nodes_flux2klein,
        "post",
        lambda url, path, payload, timeout: (
            sent.update(payload) or {"images": [], "params": {"seed": 1}}
        ),
    )
    kwargs = dict(
        prompt="p",
        negative_prompt="",
        variant="base",
        aspect_ratio="custom",
        megapixels=1.0,
        width=1024,
        height=1024,
        batch_size=3,
        seed=0,
        override_sampler=False,
        steps=20,
        cfg=5.0,
        endpoint="https://x.modal.run",
    )
    kwargs.update(overrides)
    node.generate(**kwargs)
    return sent


def test_no_reference_leaves_the_payload_text_to_image(monkeypatch):
    payload = drive_klein(monkeypatch)
    assert "reference_images" not in payload
    assert payload["batch_size"] == 3, "the batch widget still applies"


def test_a_reference_switches_the_payload_to_an_edit(monkeypatch):
    payload = drive_klein(monkeypatch, reference_image=FakeBatch(["frame"]))
    assert payload["reference_images"] == ["b64-0"]
    # A reference fixes the output size, so the batch widget cannot apply.
    assert payload["batch_size"] == 1


def test_an_empty_batch_is_not_treated_as_a_reference(monkeypatch):
    """An IMAGE input can be wired but empty; that is still text-to-image."""
    payload = drive_klein(monkeypatch, reference_image=FakeBatch([]))
    assert "reference_images" not in payload
    assert payload["batch_size"] == 3


def test_references_are_capped_at_the_servers_limit(monkeypatch):
    """The server rejects more than 4; sending 5 would fail the whole render."""
    payload = drive_klein(monkeypatch, reference_image=FakeBatch(["a", "b", "c", "d", "e", "f"]))
    assert len(payload["reference_images"]) == 4
