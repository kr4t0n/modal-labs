"""WAI-illustrious-SDXL on Modal, served as a remote ComfyUI API.

WAI-illustrious-SDXL is an Illustrious-XL finetune with native Danbooru tag
understanding. It is a single ~6.8 GB fp16 SDXL checkpoint, so it is far lighter
than the other two services here and runs comfortably on a 24 GB card.

Distribution differs too: it lives on Civitai rather than Hugging Face. The
download needs no credentials, but Civitai publishes a SHA256 and the fetch
verifies it — a third-party CDN is worth checking rather than trusting.

    modal run app.py::download_models   # one-off, ~6.8 GB into a Volume
    modal deploy app.py                 # the API
    modal serve app.py                  # the browser UI, ephemeral

Deploy-time configuration comes from the environment; see .env.example.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

import modal

HERE = Path(__file__).parent
# `add_local_python_source` resolves modules through the local interpreter, so
# both the sibling modules and the shared package at the repository root have to
# be importable no matter which directory modal is invoked from.
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from comfyui_modal import service  # noqa: E402
from comfyui_modal.service import (  # noqa: E402
    COMFY_HOST,
    COMFY_PORT,
    COMFY_URL,
    MODELS_DIR,
    UI_PORT,
)

APP_NAME = "waiillustrious-comfyui"


class CivitaiFile(NamedTuple):
    """One Civitai download, pinned by version and verified by digest."""

    model_version_id: int
    file_id: int
    destination: str
    sha256: str

    @property
    def url(self) -> str:
        # Civitai answers this with a 24-hour presigned CDN link, so the
        # redirect has to be followed at download time rather than pinned here.
        return (
            f"https://civitai.com/api/download/models/{self.model_version_id}?fileId={self.file_id}"
        )


CHECKPOINT_FILE = CivitaiFile(
    model_version_id=2883731,  # v17.0
    file_id=2763986,
    destination="checkpoints/waiIllustriousSDXL_v170.safetensors",
    sha256="f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04",
)

REQUIRED_MODELS = (CHECKPOINT_FILE.destination,)

STAGING_DIR = f"{MODELS_DIR}/.staging"

EXTRA_MODEL_PATHS_YAML = service.extra_model_paths_yaml("waiillustrious", ("checkpoints",))

# One ~6.8 GB fp16 checkpoint, so a 24 GB A10 is ample and by far the cheapest
# card that fits. SDXL is fp16 throughout, so the lack of fp8 tensor cores on
# Ampere costs nothing here. See README, "Choosing a GPU".
SETTINGS = service.Settings.from_env("WAIILLUSTRIOUS", gpu="A10")

models_volume = modal.Volume.from_name("waiillustrious-models", create_if_missing=True)

image = service.build_image(["workflow", "server", "comfyui_modal"])

app = modal.App(APP_NAME, image=image)


@app.function(volumes={MODELS_DIR: models_volume}, timeout=3600)
def download_models(force: bool = False) -> list[str]:
    """Fetch the checkpoint from Civitai, verifying its published digest.

    Idempotent by destination: an existing file is left alone unless `force`.
    A CIVITAI_TOKEN in the environment is used if present — the download is
    currently anonymous, so none is attached, but adding a Modal Secret with
    that key is all it would take should Civitai start requiring one.
    """
    import hashlib

    import httpx

    target = Path(MODELS_DIR, CHECKPOINT_FILE.destination)
    if target.is_file() and not force:
        print(f"have {CHECKPOINT_FILE.destination}, skipping")
        return [str(target)]

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(STAGING_DIR)
    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / "checkpoint.safetensors"

    headers = {}
    if token := os.environ.get("CIVITAI_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"

    digest = hashlib.sha256()
    print(f"fetching {CHECKPOINT_FILE.url} ...")
    with httpx.stream(
        "GET",
        CHECKPOINT_FILE.url,
        follow_redirects=True,
        headers=headers,
        timeout=httpx.Timeout(connect=30.0, read=None, write=None, pool=None),
    ) as response:
        response.raise_for_status()
        with staged.open("wb") as handle:
            for chunk in response.iter_bytes(4 << 20):
                handle.write(chunk)
                digest.update(chunk)

    actual = digest.hexdigest()
    if actual != CHECKPOINT_FILE.sha256:
        staged.unlink(missing_ok=True)
        raise RuntimeError(
            "checksum mismatch from Civitai: expected "
            f"{CHECKPOINT_FILE.sha256}, got {actual}. Refusing to install."
        )

    # Same Volume, so this is a rename rather than a second copy of 6.8 GB.
    os.replace(staged, target)
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    models_volume.commit()
    print(f"installed {CHECKPOINT_FILE.destination} (sha256 verified)")
    return [str(target)]


@app.cls(
    gpu=SETTINGS.gpu,
    volumes={MODELS_DIR: models_volume.read_only()},
    timeout=3600,
    startup_timeout=900,
    scaledown_window=SETTINGS.scaledown_window,
    min_containers=SETTINGS.min_containers,
    max_containers=SETTINGS.max_containers,
)
@modal.concurrent(max_inputs=SETTINGS.concurrent_inputs, target_inputs=SETTINGS.concurrent_inputs)
class WaiIllustrious:
    """A container running ComfyUI, fronted by the ASGI app in server.py."""

    @modal.enter()
    def start_comfyui(self) -> None:
        self.process = service.launch_comfyui(
            COMFY_PORT,
            COMFY_HOST,
            required_models=REQUIRED_MODELS,
            extra_paths_yaml=EXTRA_MODEL_PATHS_YAML,
        )
        service.wait_for_comfyui(self.process)

    @modal.exit()
    def stop_comfyui(self) -> None:
        service.stop_comfyui(self.process)

    @modal.asgi_app(requires_proxy_auth=SETTINGS.require_auth)
    def web(self):
        from server import create_app

        return create_app(COMFY_URL)

    @modal.method()
    def generate(self, **kwargs) -> tuple[dict, list[bytes]]:
        """Direct call path, so `modal run` can test without proxy tokens."""
        import asyncio
        import base64

        from server import GenerateRequest, comfy_client, run_generation

        async def run():
            async with comfy_client(COMFY_URL) as client:
                result = await run_generation(client, GenerateRequest(**kwargs))
                return result.params, [base64.b64decode(i.b64) for i in result.images]

        return asyncio.run(run())


@app.function(
    gpu=SETTINGS.gpu,
    volumes={MODELS_DIR: models_volume.read_only()},
    timeout=3600,
    scaledown_window=SETTINGS.scaledown_window,
    max_containers=1,
)
@modal.concurrent(max_inputs=SETTINGS.concurrent_inputs, target_inputs=SETTINGS.concurrent_inputs)
@modal.web_server(UI_PORT, startup_timeout=900, requires_proxy_auth=SETTINGS.ui_require_auth)
def ui() -> None:
    """The ComfyUI web interface, for `modal serve app.py`.

    Proxy auth is off by default because browsers cannot attach the required
    headers. Serve it ephemerally, or set WAIILLUSTRIOUS_UI_REQUIRE_AUTH=1 and
    drive it from a client that can.
    """
    service.launch_comfyui(
        UI_PORT,
        "0.0.0.0",
        required_models=REQUIRED_MODELS,
        extra_paths_yaml=EXTRA_MODEL_PATHS_YAML,
    )


@app.local_entrypoint()
def main(
    prompt: str = "1girl, solo, silver hair, red eyes, city at night, masterpiece, best quality",
    output_dir: str = "outputs",
    width: int = 832,
    height: int = 1216,
    steps: int = 28,
    seed: int = -1,
    batch_size: int = 1,
) -> None:
    """End-to-end smoke test: `modal run app.py`."""
    params, images = WaiIllustrious().generate.remote(
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        seed=None if seed < 0 else seed,
        batch_size=batch_size,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for index, data in enumerate(images):
        path = destination / f"wai_{params['seed']}_{index}.png"
        path.write_bytes(data)
        print(f"wrote {path} ({len(data) / 1e6:.2f} MB)")
    print(
        f"seed={params['seed']} steps={params['steps']} cfg={params['cfg']} "
        f"{params['width']}x{params['height']}"
    )
