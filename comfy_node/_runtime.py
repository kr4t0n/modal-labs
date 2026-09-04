"""Shared runtime for the remote-render nodes.

Everything here is model-agnostic: reading settings, talking to a Modal
endpoint, decoding the response into a ComfyUI IMAGE tensor, and mirroring the
remote sampler's progress onto the local progress bar.

This module runs inside the user's ComfyUI, so it may only import what ComfyUI
already ships: torch, numpy, PIL, requests and aiohttp.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import comfy.utils
import numpy as np
import requests
import torch
from PIL import Image

HERE = Path(__file__).parent

ASPECT_RATIOS = ["custom", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]
MAX_SEED = 0xFFFFFFFFFFFFFFFF


def _dotenv() -> dict[str, str]:
    """Minimal KEY=VALUE reader, for ComfyUI launched without a shell env."""
    path = HERE / ".env"
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def setting(name: str, default: str = "") -> str:
    """Process environment first, then a `.env` beside this package."""
    return os.environ.get(name) or _dotenv().get(name, default)


# Deliberately duplicated from comfyui_modal/cli.py, which this package cannot
# import: it is copied standalone into the user's ComfyUI, where only ComfyUI's
# own dependencies exist. A test asserts the two derivations agree.
MODAL_URL_SUFFIX = "_MODAL_URL"
WORKSPACE_VAR = "MODAL_WORKSPACE"


def derive_url(env_var: str, workspace: str) -> str:
    """The deployed URL a service *would* get, from the workspace name alone.

    Modal composes web endpoint hostnames deterministically, so one workspace
    name covers every node here and adding a service needs no new variable.

    Convenience, not a guarantee: Modal truncates or hashes hostnames past the
    DNS label limit, and a non-default Modal environment inserts a suffix. Set
    the per-service variable when the deploy output disagrees.
    """
    if not env_var.endswith(MODAL_URL_SUFFIX):
        raise RuntimeError(f"cannot derive a URL from {env_var!r}")
    slug = env_var[: -len(MODAL_URL_SUFFIX)].lower()
    return f"https://{workspace}--{slug}-comfyui-{slug}-web.modal.run"


def endpoint(override: str, env_var: str) -> str:
    url = (override or setting(env_var)).strip().rstrip("/")
    # The per-service variable wins, so pointing one node at a second deployment
    # or an ephemeral `modal serve` URL stays a one-variable override.
    #
    # The suffix check keeps a caller with a non-conforming variable name on the
    # "not configured" message below, rather than a derivation error about a
    # mechanism they were not using.
    workspace = setting(WORKSPACE_VAR).strip()
    if not url and workspace and env_var.endswith(MODAL_URL_SUFFIX):
        url = derive_url(env_var, workspace)
    if not url:
        raise RuntimeError(
            f"No endpoint configured. Set {env_var}, or set {WORKSPACE_VAR} to "
            "your Modal workspace to cover every service at once, or fill the "
            "endpoint widget with the URL printed by `modal deploy app.py`."
        )
    return url


def headers() -> dict[str, str]:
    """Modal proxy-auth headers, omitted when the endpoint is unauthenticated."""
    key, secret = setting("MODAL_KEY"), setting("MODAL_SECRET")
    if key and secret:
        return {"Modal-Key": key, "Modal-Secret": secret}
    return {}


def post(url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        response = requests.post(f"{url}{path}", json=payload, headers=headers(), timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Modal endpoint unreachable at {url}: {exc}") from exc
    if response.status_code >= 400:
        raise RuntimeError(
            f"Modal endpoint returned {response.status_code}: {response.text[:2000]}"
        )
    return response.json()


def get(url: str, path: str, params: dict[str, Any], timeout: float = 60.0) -> str:
    try:
        response = requests.get(f"{url}{path}", params=params, headers=headers(), timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Modal endpoint unreachable at {url}: {exc}") from exc
    if response.status_code >= 400:
        raise RuntimeError(f"Modal endpoint returned {response.status_code}: {response.text[:500]}")
    return response.text


def to_tensor(images: list[dict[str, Any]]) -> torch.Tensor:
    """Decode the response into ComfyUI's [B, H, W, C] float32 image batch."""
    frames = []
    for record in images:
        pil = Image.open(io.BytesIO(base64.b64decode(record["b64"]))).convert("RGB")
        frames.append(torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0))
    return torch.stack(frames)


class ProgressMirror:
    """Mirror a remote render's progress onto this node's local progress bar.

    ComfyUI addresses progress events to the client id that submitted the
    prompt, so subscribing to the deployment's websocket with the id we passed
    to `/generate` yields the remote sampler's per-step progress. Feeding it to
    `comfy.utils.ProgressBar` is what the local UI draws, so a render happening
    on a Modal GPU looks like a local one.

    Progress is cosmetic and must never break a render: every failure in here is
    swallowed and the generation call carries on regardless.

    On clustered installs it also has a load-bearing side effect. Each update
    makes the local server push a frame to the browser, so the tab's websocket
    never sits idle for the length of a render — an ingress that idle-timeouts
    websockets would otherwise drop it mid-job.
    """

    # Long enough to cover the socket handshake, short enough that an
    # unreachable websocket does not visibly stall the render.
    CONNECT_TIMEOUT_S = 5.0

    # How often the receive loop wakes to notice it has been asked to stop.
    # This is dead time on every render, since __exit__ runs once the image is
    # already in hand, so it is kept short.
    POLL_INTERVAL_S = 0.25

    def __init__(self, url: str, client_id: str, node_id: str | None) -> None:
        stream_url = url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        self._ws_url = f"{stream_url}/ws?clientId={client_id}"
        self._node_id = node_id
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ProgressMirror:
        self._thread = threading.Thread(target=self._run, name="modal-progress", daemon=True)
        self._thread.start()
        # Subscribe before the prompt is queued, or the first steps are missed.
        self._connected.wait(self.CONNECT_TIMEOUT_S)
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return False

    def _run(self) -> None:
        try:
            asyncio.run(self._pump())
        except Exception:
            pass
        finally:
            # Unblock __enter__ even if the socket never came up.
            self._connected.set()

    async def _pump(self) -> None:
        # aiohttp is one of ComfyUI's own dependencies, so it is always present;
        # imported here so any surprise stays inside the swallowed thread.
        import aiohttp

        closing = {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        }
        async with (
            aiohttp.ClientSession(headers=headers()) as session,
            session.ws_connect(self._ws_url, heartbeat=20) as socket,
        ):
            self._connected.set()
            bar = None
            while not self._stop.is_set():
                try:
                    message = await asyncio.wait_for(socket.receive(), timeout=self.POLL_INTERVAL_S)
                except TimeoutError:
                    continue
                if message.type in closing:
                    return
                if message.type is aiohttp.WSMsgType.TEXT:
                    bar = self._apply(json.loads(message.data), bar)

    def _apply(self, event: dict[str, Any], bar):
        """Translate one `progress` event into a local progress-bar update."""
        if event.get("type") != "progress":
            return bar
        data = event.get("data") or {}
        value, total = data.get("value"), data.get("max")
        if not isinstance(value, int) or not isinstance(total, int) or total <= 0:
            return bar
        if bar is None or bar.total != total:
            bar = self._new_bar(total)
        bar.update_absolute(value, total)
        return bar

    def _new_bar(self, total: int):
        """`node_id` binds the bar to this node; older ComfyUI lacks the kwarg.

        Without it the bar still draws, just against the graph rather than the
        node — better than the silent no-op a TypeError in this thread becomes.
        """
        try:
            return comfy.utils.ProgressBar(total, node_id=self._node_id)
        except TypeError:
            return comfy.utils.ProgressBar(total)


def common_geometry_inputs(
    *, default_side: int = 1024, default_megapixels: float = 1.0
) -> dict[str, Any]:
    """The width/height/aspect-ratio/seed widgets every render node shares.

    The widgets always send a value, so a model whose native resolution is not
    1 MP needs its defaults moved here rather than left to the server.
    """
    return {
        "aspect_ratio": (
            ASPECT_RATIOS,
            {"default": "1:1", "tooltip": "'custom' uses the width/height widgets instead."},
        ),
        "megapixels": (
            "FLOAT",
            {"default": default_megapixels, "min": 0.1, "max": 4.0, "step": 0.1},
        ),
        "width": ("INT", {"default": default_side, "min": 256, "max": 2048, "step": 16}),
        "height": ("INT", {"default": default_side, "min": 256, "max": 2048, "step": 16}),
        "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
        "seed": (
            "INT",
            {"default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True},
        ),
    }


def endpoint_inputs(env_var: str) -> dict[str, Any]:
    return {
        "endpoint": ("STRING", {"default": "", "tooltip": f"Overrides {env_var}."}),
        "timeout_s": ("FLOAT", {"default": 900.0, "min": 30.0, "max": 3600.0}),
    }


def geometry_payload(aspect_ratio: str, megapixels: float, width: int, height: int) -> dict:
    payload: dict[str, Any] = {"width": width, "height": height}
    if aspect_ratio != "custom":
        payload["aspect_ratio"] = aspect_ratio
        payload["megapixels"] = megapixels
    return payload
