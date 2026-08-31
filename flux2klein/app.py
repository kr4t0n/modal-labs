"""FLUX.2 [klein] 9B on Modal, served as a remote ComfyUI API.

FLUX.2 klein is Black Forest Labs' compact FLUX.2: a 9B transformer with a
Qwen3-8B text encoder, shipped in two flavours — an undistilled `base` that
responds to CFG and negative prompts, and a 4-step guidance-`distilled` build.
Both are supported by ComfyUI core.

Same shape as the ideogram4 service: a real headless ComfyUI runs on the GPU and
a thin ASGI layer fronts it, so the URL speaks the ComfyUI protocol (`/prompt`,
`/history`, `/view`, `/ws`, the web UI) and also answers a typed `/generate`
contract.

    modal secret create huggingface HF_TOKEN=hf_...   # the transformers are gated
    modal run app.py::download_models                 # one-off, ~28 GB
    modal deploy app.py                               # the API
    modal serve app.py                                # the browser UI, ephemeral

Deploy-time configuration comes from the environment; see .env.example.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import modal

HERE = Path(__file__).parent
# `add_local_python_source` resolves modules through the local interpreter, so
# the sibling modules have to be importable no matter where modal is invoked.
sys.path.insert(0, str(HERE))

APP_NAME = "flux2klein-comfyui"

# Flux2Scheduler and EmptyFlux2LatentImage are what workflow.py needs. Pinned
# to the same release the ideogram4 service runs.
COMFYUI_REF = "v0.34.2"

COMFYUI_DIR = "/root/ComfyUI"
MODELS_DIR = "/root/models"

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
UI_PORT = 8000


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", ""}


# --- Deploy-time configuration ---------------------------------------------
# Only one transformer is resident at a time here, so the working set is ~18.5 GB
# — materially smaller than the ideogram4 service, and a better candidate for a
# cheaper GPU. Compare cost per image, not per hour. See README, "Choosing a GPU".
GPU = os.environ.get("FLUX2KLEIN_GPU", "H100")
MIN_CONTAINERS = int(os.environ.get("FLUX2KLEIN_MIN_CONTAINERS", "0"))
# Defaults to one so the ComfyUI queue, /history and /view stay coherent: those
# are per-container state, and a second replica would answer for prompts it
# never ran. Raise it only if your client submits and polls in one request.
MAX_CONTAINERS = int(os.environ.get("FLUX2KLEIN_MAX_CONTAINERS", "1"))
SCALEDOWN_WINDOW = int(os.environ.get("FLUX2KLEIN_SCALEDOWN_WINDOW", "300"))
# Concurrency is about not blocking the proxy: ComfyUI serialises sampling
# itself, but progress polls and /view fetches must get through mid-render.
CONCURRENT_INPUTS = int(os.environ.get("FLUX2KLEIN_CONCURRENT_INPUTS", "20"))
REQUIRE_AUTH = _flag("FLUX2KLEIN_REQUIRE_AUTH", "1")
UI_REQUIRE_AUTH = _flag("FLUX2KLEIN_UI_REQUIRE_AUTH", "0")

# --- Weights ----------------------------------------------------------------
# Unlike the ideogram4 service these come from four repos whose internal layouts
# do not match ComfyUI's, so every file carries an explicit destination.
HF_SECRET_NAME = os.environ.get("FLUX2KLEIN_HF_SECRET", "huggingface")


class ModelFile(NamedTuple):
    repo_id: str
    filename: str
    destination: str
    gated: bool


MODEL_FILES = (
    ModelFile(
        "black-forest-labs/FLUX.2-klein-base-9b-fp8",
        "flux-2-klein-base-9b-fp8.safetensors",
        "diffusion_models/flux-2-klein-base-9b-fp8.safetensors",
        gated=True,
    ),
    ModelFile(
        "black-forest-labs/FLUX.2-klein-9b-fp8",
        "flux-2-klein-9b-fp8.safetensors",
        "diffusion_models/flux-2-klein-9b-fp8.safetensors",
        gated=True,
    ),
    ModelFile(
        "Comfy-Org/flux2-klein-9B",
        "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
        "text_encoders/qwen_3_8b_fp8mixed.safetensors",
        gated=False,
    ),
    ModelFile(
        "black-forest-labs/FLUX.2-small-decoder",
        "full_encoder_small_decoder.safetensors",
        "vae/full_encoder_small_decoder.safetensors",
        gated=False,
    ),
)

# Downloads land here first and are then renamed into place. Same Volume, so the
# move is a rename rather than a second copy of 9 GB.
STAGING_DIR = f"{MODELS_DIR}/.staging"

# Search paths rather than a bind-mount over ComfyUI/models, which would hide
# the configs the repo ships there. Written at container start, not baked into
# the image: `run_commands` entries become Dockerfile RUN lines, and embedding a
# multi-line document in one is a parse error.
EXTRA_MODEL_PATHS_YAML = f"""flux2klein:
  base_path: {MODELS_DIR}
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
"""

models_volume = modal.Volume.from_name("flux2klein-models", create_if_missing=True)


def _single_line(*commands: str) -> tuple[str, ...]:
    """Reject multi-line build commands at import time.

    Each `run_commands` entry becomes one Dockerfile RUN line. Interpolating a
    multi-line value into one produces a Dockerfile that fails to parse, and the
    error surfaces minutes into a remote build rather than here.
    """
    for command in commands:
        if "\n" in command:
            raise ValueError(f"run_commands entry spans multiple lines: {command!r}")
    return commands


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.13.0", "torchvision==0.28.0", "torchaudio==2.11.0")
    .run_commands(
        *_single_line(
            f"git clone --depth 1 --branch {COMFYUI_REF}"
            f" https://github.com/comfyanonymous/ComfyUI.git {COMFYUI_DIR}",
            f"pip install --no-cache-dir -r {COMFYUI_DIR}/requirements.txt",
        )
    )
    .pip_install(
        "huggingface_hub[hf_transfer]==1.29.0",
        "fastapi==0.141.1",
        "httpx==0.28.1",
        "websockets==17.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("workflow", "server")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={MODELS_DIR: models_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])],
)
def download_models(force: bool = False) -> list[str]:
    """Populate the weights Volume.

    Idempotent by destination: an existing file is left alone unless `force`.
    The two transformers are gated, so the Hugging Face account behind HF_TOKEN
    must have accepted the FLUX.2 licence first, or the download 401s.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    written = []
    for model in MODEL_FILES:
        target = Path(MODELS_DIR, model.destination)
        if target.is_file() and not force:
            print(f"have {model.destination}, skipping")
            written.append(str(target))
            continue

        print(f"fetching {model.repo_id}/{model.filename} ...")
        try:
            staged = hf_hub_download(
                repo_id=model.repo_id, filename=model.filename, local_dir=STAGING_DIR
            )
        except GatedRepoError as exc:
            raise RuntimeError(
                f"{model.repo_id} is gated. Accept the licence at "
                f"https://huggingface.co/{model.repo_id} using the account that owns "
                "HF_TOKEN, then re-run."
            ) from exc

        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, target)
        written.append(str(target))

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    models_volume.commit()
    return written


def _assert_models_present() -> None:
    missing = [m.destination for m in MODEL_FILES if not Path(MODELS_DIR, m.destination).is_file()]
    if missing:
        raise RuntimeError(
            "weights Volume is missing "
            + ", ".join(missing)
            + " — run `modal run app.py::download_models` first"
        )


def _launch_comfyui(port: int, listen: str) -> subprocess.Popen:
    _assert_models_present()
    # ComfyUI loads this from its own directory during startup.
    Path(COMFYUI_DIR, "extra_model_paths.yaml").write_text(EXTRA_MODEL_PATHS_YAML)
    return subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "--listen",
            listen,
            "--port",
            str(port),
            "--disable-auto-launch",
        ],
        cwd=COMFYUI_DIR,
    )


def _wait_for_comfyui(url: str, process: subprocess.Popen, timeout: float = 600.0) -> None:
    """Block until ComfyUI answers, failing fast if the process dies."""
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (code := process.poll()) is not None:
            raise RuntimeError(f"ComfyUI exited during startup with code {code}")
        try:
            if httpx.get(f"{url}/system_stats", timeout=5.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"ComfyUI did not become ready within {timeout}s")


@app.cls(
    gpu=GPU,
    volumes={MODELS_DIR: models_volume.read_only()},
    timeout=3600,
    startup_timeout=900,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=CONCURRENT_INPUTS, target_inputs=CONCURRENT_INPUTS)
class Flux2Klein:
    """A container running ComfyUI, fronted by the ASGI app in server.py."""

    @modal.enter()
    def start_comfyui(self) -> None:
        self.process = _launch_comfyui(COMFY_PORT, COMFY_HOST)
        _wait_for_comfyui(COMFY_URL, self.process)

    @modal.exit()
    def stop_comfyui(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()

    @modal.asgi_app(requires_proxy_auth=REQUIRE_AUTH)
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
    gpu=GPU,
    volumes={MODELS_DIR: models_volume.read_only()},
    timeout=3600,
    scaledown_window=SCALEDOWN_WINDOW,
    max_containers=1,
)
@modal.concurrent(max_inputs=CONCURRENT_INPUTS, target_inputs=CONCURRENT_INPUTS)
@modal.web_server(UI_PORT, startup_timeout=900, requires_proxy_auth=UI_REQUIRE_AUTH)
def ui() -> None:
    """The ComfyUI web interface, for `modal serve app.py`.

    Proxy auth is off by default because browsers cannot attach the required
    headers. Serve it ephemerally, or set FLUX2KLEIN_UI_REQUIRE_AUTH=1 and drive
    it from a client that can.
    """
    _launch_comfyui(UI_PORT, "0.0.0.0")


@app.local_entrypoint()
def main(
    prompt: str = "a vintage motorcycle parked in front of a retro diner at sunset",
    output_dir: str = "outputs",
    variant: str = "base",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    batch_size: int = 1,
) -> None:
    """End-to-end smoke test: `modal run app.py`."""
    params, images = Flux2Klein().generate.remote(
        prompt=prompt,
        variant=variant,
        width=width,
        height=height,
        seed=None if seed < 0 else seed,
        batch_size=batch_size,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for index, data in enumerate(images):
        path = destination / f"flux2klein_{params['seed']}_{index}.png"
        path.write_bytes(data)
        print(f"wrote {path} ({len(data) / 1e6:.2f} MB)")
    print(
        f"seed={params['seed']} variant={params['variant']} steps={params['steps']} "
        f"cfg={params['cfg']} {params['width']}x{params['height']}"
    )
