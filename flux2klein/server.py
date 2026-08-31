"""FastAPI front end for the ComfyUI process running inside the same container.

One URL exposes two surfaces:

* A typed contract (``/generate``, ``/workflow``, ``/variants``) that
  hides the graph from callers. The bundled ComfyUI node and ``client.py`` use
  this.
* A transparent reverse proxy for everything else, so the URL *is* a ComfyUI
  server: ``/prompt``, ``/history``, ``/view``, ``/object_info``,
  ``/upload/image``, the ``/ws`` progress socket and the web UI all behave
  exactly as they do against a local install.

The proxy is deliberately dumb — it rewrites nothing but hop-by-hop headers —
so a ComfyUI client cannot tell it apart from the real thing.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import time
import uuid
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

import workflow

# Headers that describe a single hop and so must not cross the proxy.
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

# Additionally dropped on the way up: httpx sets `host` for the new connection,
# and the body is re-streamed so any inbound length no longer applies.
REQUEST_ONLY_DROPPED_HEADERS = HOP_BY_HOP_HEADERS | {"host", "content-length"}

# `content-encoding` and `content-length` are deliberately *not* dropped on the
# way down: the response body is forwarded raw (still compressed), so stripping
# the encoding header would hand the client gzip labelled as plain text.

PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

# ComfyUI mirrors every route under /api for its own frontend, so both spellings
# of the progress socket have to be handled.
WEBSOCKET_PATHS = ("/ws", "/api/ws")


class GenerateRequest(BaseModel):
    """Everything the FLUX.2 klein graph needs, with the variant defaults applied."""

    prompt: str = Field(description="Natural-language prompt.")
    negative_prompt: str = Field(
        default="",
        description="Only meaningful on the base variant; the distilled one ignores it.",
    )
    variant: Literal["base", "distilled"] = Field(
        default=workflow.DEFAULT_VARIANT,
        description=(
            "'base' is undistilled (20 steps, cfg 5). 'distilled' is a 4-step "
            "guidance-distilled model that ignores cfg and negative prompts."
        ),
    )

    width: int = Field(default=1024, ge=workflow.MIN_SIDE, le=workflow.MAX_SIDE)
    height: int = Field(default=1024, ge=workflow.MIN_SIDE, le=workflow.MAX_SIDE)
    aspect_ratio: str | None = Field(
        default=None,
        description="If set, overrides width/height using megapixels as the budget.",
    )
    megapixels: float = Field(default=1.0, gt=0, le=8.0)

    seed: int | None = Field(default=None, description="Random when omitted.")
    batch_size: int = Field(default=1, ge=1, le=8)

    steps: int | None = Field(default=None, ge=1, le=200, description="Overrides the variant.")
    cfg: float | None = Field(default=None, ge=0.0, le=100.0, description="Overrides the variant.")
    sampler_name: str = "euler"

    filename_prefix: str = "flux2-klein"
    timeout_s: float = Field(default=900.0, gt=0, le=3600.0)

    client_id: str | None = Field(
        default=None,
        description=(
            "Forwarded to ComfyUI as the prompt's client id. Supply one and "
            "subscribe to /ws?clientId=<id> to receive this render's progress "
            "events; a fresh id is generated when omitted."
        ),
    )

    def dimensions(self) -> tuple[int, int]:
        if self.aspect_ratio is None:
            return self.width, self.height
        try:
            return workflow.resolution_for(self.aspect_ratio, self.megapixels)
        except workflow.WorkflowError as exc:
            raise HTTPException(422, str(exc)) from exc


class GeneratedImage(BaseModel):
    filename: str
    subfolder: str
    type: str
    content_type: str
    b64: str


class GenerateResponse(BaseModel):
    prompt_id: str
    duration_s: float
    params: dict[str, Any]
    images: list[GeneratedImage]


def _resolve(request: GenerateRequest) -> workflow.GenerationParams:
    width, height = request.dimensions()
    try:
        return workflow.resolve_params(
            request.prompt,
            negative_prompt=request.negative_prompt,
            variant=request.variant,
            width=width,
            height=height,
            seed=request.seed,
            batch_size=request.batch_size,
            steps=request.steps,
            cfg=request.cfg,
            sampler_name=request.sampler_name,
            filename_prefix=request.filename_prefix,
        )
    except workflow.WorkflowError as exc:
        raise HTTPException(422, str(exc)) from exc


async def _submit(
    client: httpx.AsyncClient, graph: dict[str, Any], client_id: str | None = None
) -> str:
    """Queue the graph and return its prompt id, surfacing validation errors.

    ComfyUI addresses progress events to the submitting client id, so passing
    the caller's through is what lets a remote node mirror the render.
    """
    payload = {"prompt": graph, "client_id": client_id or uuid.uuid4().hex}
    response = await client.post("/prompt", json=payload)
    if response.status_code >= 400:
        # ComfyUI answers 400 with {"error": ..., "node_errors": {...}}; passing
        # it through verbatim is far more useful than a generic 502.
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise HTTPException(response.status_code, detail)
    return response.json()["prompt_id"]


async def _await_history(
    client: httpx.AsyncClient, prompt_id: str, timeout_s: float
) -> dict[str, Any]:
    """Poll until the prompt leaves the queue, then return its history entry."""
    deadline = time.monotonic() + timeout_s
    delay = 0.25
    while True:
        response = await client.get(f"/history/{prompt_id}")
        response.raise_for_status()
        entry = response.json().get(prompt_id)
        if entry is not None:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise HTTPException(500, {"comfyui_error": status.get("messages", [])})
            if status.get("completed", True):
                return entry
        if time.monotonic() > deadline:
            raise HTTPException(504, f"prompt {prompt_id} did not finish in {timeout_s}s")
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 2.0)


async def _fetch_images(client: httpx.AsyncClient, entry: dict[str, Any]) -> list[GeneratedImage]:
    outputs = entry.get("outputs", {}).get(workflow.OUTPUT_NODE_ID, {})
    images = []
    for record in outputs.get("images", []):
        response = await client.get(
            "/view",
            params={
                "filename": record["filename"],
                "subfolder": record.get("subfolder", ""),
                "type": record.get("type", "output"),
            },
        )
        response.raise_for_status()
        images.append(
            GeneratedImage(
                filename=record["filename"],
                subfolder=record.get("subfolder", ""),
                type=record.get("type", "output"),
                content_type=response.headers.get("content-type", "image/png"),
                b64=base64.b64encode(response.content).decode("ascii"),
            )
        )
    if not images:
        raise HTTPException(500, "ComfyUI finished but produced no images")
    return images


def comfy_client(comfy_url: str) -> httpx.AsyncClient:
    """An HTTP client for ComfyUI with read timeouts disabled.

    A 2K/48-step render takes minutes; any read timeout here would abort a job
    that is progressing normally.
    """
    return httpx.AsyncClient(
        base_url=comfy_url,
        timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
    )


async def run_generation(client: httpx.AsyncClient, request: GenerateRequest) -> GenerateResponse:
    """Queue the Ideogram 4 graph, wait for it, and collect the images."""
    params = _resolve(request)
    started = time.monotonic()
    prompt_id = await _submit(client, workflow.build_workflow(params), request.client_id)
    entry = await _await_history(client, prompt_id, request.timeout_s)
    images = await _fetch_images(client, entry)
    return GenerateResponse(
        prompt_id=prompt_id,
        duration_s=round(time.monotonic() - started, 2),
        params=params.as_dict(),
        images=images,
    )


def create_app(comfy_url: str) -> FastAPI:
    """Build the ASGI app in front of a ComfyUI server at ``comfy_url``."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with comfy_client(comfy_url) as client:
            app.state.http = client
            yield

    web_app = FastAPI(
        title="FLUX.2 klein 9B on ComfyUI",
        description="A ComfyUI server with the FLUX.2 klein 9B weights, plus a typed generate endpoint.",
        lifespan=lifespan,
    )

    @web_app.get("/health")
    async def health() -> dict[str, Any]:
        client: httpx.AsyncClient = web_app.state.http
        try:
            response = await client.get("/system_stats", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(503, f"ComfyUI not reachable: {exc}") from exc
        return {"status": "ok", "system_stats": response.json()}

    @web_app.get("/variants")
    async def variants() -> dict[str, Any]:
        return {
            "variants": {
                name: {
                    "checkpoint": spec.checkpoint,
                    "steps": spec.steps,
                    "cfg": spec.cfg,
                    "description": spec.description,
                }
                for name, spec in workflow.VARIANTS.items()
            },
            "default": workflow.DEFAULT_VARIANT,
            "aspect_ratios": sorted(workflow.ASPECT_RATIOS),
            "resolution": {
                "min": workflow.MIN_SIDE,
                "max": workflow.MAX_SIDE,
                "multiple_of": workflow.SIDE_MULTIPLE,
            },
        }

    @web_app.post("/workflow")
    async def build(request: GenerateRequest) -> dict[str, Any]:
        """Return the API-format graph without running it."""
        params = _resolve(request)
        return {"params": params.as_dict(), "workflow": workflow.build_workflow(params)}

    @web_app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        return await run_generation(web_app.state.http, request)

    @web_app.post("/generate/image")
    async def generate_image(request: GenerateRequest) -> Response:
        """Same as /generate but returns the first image as raw bytes."""
        result = await generate(request)
        first = result.images[0]
        return Response(
            base64.b64decode(first.b64),
            media_type=first.content_type,
            headers={
                "X-Ideogram4-Seed": str(result.params["seed"]),
                "X-Ideogram4-Prompt-Id": result.prompt_id,
            },
        )

    async def _proxy_websocket(client_ws: WebSocket, path: str) -> None:
        await client_ws.accept()
        query = client_ws.url.query
        target = comfy_url.replace("http://", "ws://", 1) + path
        if query:
            target = f"{target}?{query}"
        try:
            async with ws_connect(target, max_size=None) as upstream:

                async def to_upstream() -> None:
                    while True:
                        message = await client_ws.receive()
                        if message["type"] == "websocket.disconnect":
                            return
                        if (text := message.get("text")) is not None:
                            await upstream.send(text)
                        elif (data := message.get("bytes")) is not None:
                            await upstream.send(data)

                async def to_client() -> None:
                    async for message in upstream:
                        if isinstance(message, str):
                            await client_ws.send_text(message)
                        else:
                            await client_ws.send_bytes(message)

                done, pending = await asyncio.wait(
                    [asyncio.create_task(to_upstream()), asyncio.create_task(to_client())],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    task.result()
        except (ConnectionClosed, WebSocketDisconnect):
            pass
        finally:
            with contextlib.suppress(RuntimeError):
                await client_ws.close()

    def _websocket_handler(path: str):
        async def handler(websocket: WebSocket) -> None:
            await _proxy_websocket(websocket, path)

        return handler

    for ws_path in WEBSOCKET_PATHS:
        # Starlette's raw route rather than FastAPI's: the handler takes the
        # socket only, with no dependency-injection pass over its signature.
        web_app.router.add_websocket_route(ws_path, _websocket_handler(ws_path))

    # Registered last: FastAPI matches routes in order, so the typed endpoints
    # above win and everything else falls through to ComfyUI untouched.
    @web_app.api_route("/{path:path}", methods=PROXY_METHODS, include_in_schema=False)
    async def proxy(path: str, request: Request) -> Response:
        client: httpx.AsyncClient = request.app.state.http
        url = httpx.URL(f"{comfy_url}/{path}")
        if request.url.query:
            url = url.copy_with(query=request.url.query.encode("utf-8"))
        headers = [
            (key, value)
            for key, value in request.headers.raw
            if key.decode("latin-1").lower() not in REQUEST_ONLY_DROPPED_HEADERS
        ]
        # Only stream a body when the client actually sent one: attaching an
        # iterator to a plain GET makes httpx declare a chunked body, which
        # aiohttp rejects.
        has_body = "content-length" in request.headers or "transfer-encoding" in request.headers
        upstream_request = client.build_request(
            request.method,
            url,
            headers=headers,
            content=request.stream() if has_body else None,
        )
        upstream = await client.send(upstream_request, stream=True)
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers={
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS
            },
            background=BackgroundTask(upstream.aclose),
        )

    return web_app
