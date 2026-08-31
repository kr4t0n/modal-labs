"""ComfyUI node that renders on a remote FLUX.2 klein 9B deployment.

Drop this package into `ComfyUI/custom_nodes/` on the machine you actually work
on. The node holds no weights: it posts to the Modal endpoint's `/generate`
contract and returns the decoded image as a normal IMAGE tensor, so it composes
with upscalers, savers and everything else in a local workflow.

Credentials never enter the workflow JSON. The endpoint URL and Modal proxy
token are read from the environment, falling back to a `.env` file beside this
module — see .env.example.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import comfy.utils
import numpy as np
import requests
import torch
from PIL import Image

HERE = Path(__file__).parent

ASPECT_RATIOS = ["custom", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]
VARIANTS = ["base", "distilled"]
MAX_SEED = 0xFFFFFFFFFFFFFFFF

CATEGORY = "FLUX.2 klein (Modal)"


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


def _setting(name: str, default: str = "") -> str:
    return os.environ.get(name) or _dotenv().get(name, default)


def _endpoint(override: str) -> str:
    url = (override or _setting("FLUX2KLEIN_MODAL_URL")).strip().rstrip("/")
    if not url:
        raise RuntimeError(
            "No endpoint configured. Set FLUX2KLEIN_MODAL_URL (or fill the "
            "endpoint widget) to the URL printed by `modal deploy app.py`."
        )
    return url


def _headers() -> dict[str, str]:
    """Modal proxy-auth headers, omitted when the endpoint is unauthenticated."""
    key, secret = _setting("MODAL_KEY"), _setting("MODAL_SECRET")
    if key and secret:
        return {"Modal-Key": key, "Modal-Secret": secret}
    return {}


def _post(url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        response = requests.post(f"{url}{path}", json=payload, headers=_headers(), timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"FLUX.2 klein endpoint unreachable at {url}: {exc}") from exc
    if response.status_code >= 400:
        raise RuntimeError(
            f"FLUX.2 klein endpoint returned {response.status_code}: {response.text[:2000]}"
        )
    return response.json()


def _to_tensor(images: list[dict[str, Any]]) -> torch.Tensor:
    """Decode the response into ComfyUI's [B, H, W, C] float32 image batch."""
    frames = []
    for record in images:
        pil = Image.open(io.BytesIO(base64.b64decode(record["b64"]))).convert("RGB")
        frames.append(torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0))
    return torch.stack(frames)


class _ProgressMirror:
    """Mirror a remote render's progress onto this node's local progress bar.

    ComfyUI addresses progress events to the client id that submitted the
    prompt, so subscribing to the deployment's websocket with the id we passed
    to `/generate` yields the remote sampler's per-step progress. Feeding it to
    `comfy.utils.ProgressBar` is what the local UI draws, so a render happening
    on a Modal GPU looks like a local one.

    Progress is cosmetic and must never break a render: every failure in here
    is swallowed and the generation call carries on regardless.

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

    def __enter__(self) -> _ProgressMirror:
        self._thread = threading.Thread(target=self._run, name="flux2klein-progress", daemon=True)
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
            aiohttp.ClientSession(headers=_headers()) as session,
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


class Flux2KleinModal:
    """Text to image on the remote FLUX.2 klein 9B deployment."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a vintage motorcycle parked in front of a retro diner at sunset",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Ignored by the distilled variant, which is guidance-distilled.",
                    },
                ),
                "variant": (
                    VARIANTS,
                    {
                        "default": "base",
                        "tooltip": "base = 20 steps at cfg 5. distilled = 4 steps, ignores cfg.",
                    },
                ),
                "aspect_ratio": (
                    ASPECT_RATIOS,
                    {
                        "default": "1:1",
                        "tooltip": "'custom' uses the width/height widgets instead.",
                    },
                ),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_SEED, "control_after_generate": True},
                ),
                "override_sampler": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Off means the variant's own steps/cfg are used.",
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "endpoint": (
                    "STRING",
                    {"default": "", "tooltip": "Overrides FLUX2KLEIN_MODAL_URL."},
                ),
                "timeout_s": ("FLOAT", {"default": 900.0, "min": 30.0, "max": 3600.0}),
            },
            # Lets the progress bar attach to this node rather than the graph.
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "info")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Render with FLUX.2 klein 9B on a Modal-hosted ComfyUI and return the image."

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        variant: str,
        aspect_ratio: str,
        megapixels: float,
        width: int,
        height: int,
        batch_size: int,
        seed: int,
        override_sampler: bool,
        steps: int,
        cfg: float,
        endpoint: str = "",
        timeout_s: float = 900.0,
        unique_id: str | None = None,
    ):
        url = _endpoint(endpoint)
        client_id = uuid.uuid4().hex
        payload: dict[str, Any] = {
            "client_id": client_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "variant": variant,
            "width": width,
            "height": height,
            "batch_size": batch_size,
            "seed": seed,
            # Give the remote a little slack so it reports the timeout rather
            # than the socket dying underneath us.
            "timeout_s": max(timeout_s - 15.0, 30.0),
        }
        if aspect_ratio != "custom":
            payload["aspect_ratio"] = aspect_ratio
            payload["megapixels"] = megapixels
        # Left out, the server falls back to the variant's tuned steps/cfg.
        if override_sampler:
            payload["steps"] = steps
            payload["cfg"] = cfg

        with _ProgressMirror(url, client_id, unique_id):
            result = _post(url, "/generate", payload, timeout_s)

        params = result.get("params", {})
        notes = ""
        if variant == "distilled" and negative_prompt.strip():
            notes = " (negative prompt ignored: distilled variant)"
        info = (
            f"{params.get('variant')} seed={params.get('seed')} "
            f"steps={params.get('steps')} cfg={params.get('cfg')} "
            f"{params.get('width')}x{params.get('height')} "
            f"in {result.get('duration_s')}s{notes}"
        )
        return (_to_tensor(result["images"]), int(params.get("seed", seed)), info)


NODE_CLASS_MAPPINGS = {"Flux2KleinModal": Flux2KleinModal}

NODE_DISPLAY_NAME_MAPPINGS = {"Flux2KleinModal": "FLUX.2 klein (Modal)"}
