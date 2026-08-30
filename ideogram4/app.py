"""Ideogram 4 on Modal, served as a remote ComfyUI API.

Ideogram 4 is a 9.3B single-stream diffusion transformer with open weights and
first-class support in ComfyUI core. Rather than reimplement its sampling
schedule, this deployment runs a real headless ComfyUI on a Modal GPU and puts
a thin ASGI app in front of it. The resulting URL behaves like a ComfyUI server
(`/prompt`, `/history`, `/view`, `/ws`, the web UI) and additionally answers a
typed `/generate` contract that the bundled custom node calls.

Two entrypoints:

    modal run app.py::download_models   # one-off, ~30 GB into a Volume
    modal deploy app.py                 # the API
    modal serve app.py                  # the browser UI, ephemeral

Deploy-time configuration comes from the environment; see .env.example.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
# `add_local_python_source` resolves modules through the local interpreter, so
# the sibling modules have to be importable no matter where modal is invoked.
sys.path.insert(0, str(HERE))

APP_NAME = "ideogram4-comfyui"

# Ideogram4Scheduler, DualModelGuider and CFGOverride all landed in ComfyUI
# 0.23.0; the graph in workflow.py will not validate against anything older.
COMFYUI_REF = "v0.34.2"

COMFYUI_DIR = "/root/ComfyUI"
MODELS_DIR = "/root/models"
CAPTION_TEMPLATE_PATH = "/root/assets/magic_prompt_template.txt"

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
UI_PORT = 8000


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", ""}


# --- Deploy-time configuration ---------------------------------------------
# 29.5 GB of fp8 weights total; ~18.9 GB is resident during sampling, once the
# text encoder is offloaded. A 48 GB L40S therefore holds the lot and is roughly
# half H100's hourly rate — but it is also slower, so compare cost per image
# rather than per hour before switching. See README, "Choosing a GPU".
GPU = os.environ.get("IDEOGRAM4_GPU", "H100")
MIN_CONTAINERS = int(os.environ.get("IDEOGRAM4_MIN_CONTAINERS", "0"))
# Defaults to one so the ComfyUI queue, /history and /view stay coherent: those
# are per-container state, and a second replica would answer for prompts it
# never ran. Raise it only if your client submits and polls in one request.
MAX_CONTAINERS = int(os.environ.get("IDEOGRAM4_MAX_CONTAINERS", "1"))
SCALEDOWN_WINDOW = int(os.environ.get("IDEOGRAM4_SCALEDOWN_WINDOW", "300"))
# Concurrency is about not blocking the proxy: ComfyUI serialises sampling
# itself, but progress polls and /view fetches must get through mid-render.
CONCURRENT_INPUTS = int(os.environ.get("IDEOGRAM4_CONCURRENT_INPUTS", "20"))
REQUIRE_AUTH = _flag("IDEOGRAM4_REQUIRE_AUTH", "1")
UI_REQUIRE_AUTH = _flag("IDEOGRAM4_UI_REQUIRE_AUTH", "0")

# --- Weights ----------------------------------------------------------------
# Comfy-Org's mirror of the Ideogram 4 release, pre-split into ComfyUI's layout.
# Downloading with `local_dir=MODELS_DIR` reproduces the repo's subdirectories,
# which is exactly the layout extra_model_paths.yaml points at.
MODEL_REPO = "Comfy-Org/Ideogram-4"
MODEL_FILES = (
    "diffusion_models/ideogram4_fp8_scaled.safetensors",
    "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
    "text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
    "vae/flux2-vae.safetensors",
)

# Search paths rather than a bind-mount over ComfyUI/models, which would hide
# the configs the repo ships there. Written at container start, not baked into
# the image: `run_commands` entries become Dockerfile RUN lines, and embedding a
# multi-line document in one is a parse error.
EXTRA_MODEL_PATHS_YAML = f"""ideogram4:
  base_path: {MODELS_DIR}
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
"""

models_volume = modal.Volume.from_name("ideogram4-models", create_if_missing=True)


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
    .add_local_file(HERE / "assets" / "magic_prompt_template.txt", CAPTION_TEMPLATE_PATH)
    .add_local_python_source("workflow", "server")
)

app = modal.App(APP_NAME, image=image)


@app.function(volumes={MODELS_DIR: models_volume}, timeout=3600)
def download_models() -> list[str]:
    """Populate the weights Volume. Idempotent; re-running verifies checksums."""
    from huggingface_hub import hf_hub_download

    paths = []
    for filename in MODEL_FILES:
        print(f"fetching {filename} ...")
        paths.append(hf_hub_download(repo_id=MODEL_REPO, filename=filename, local_dir=MODELS_DIR))
    models_volume.commit()
    return paths


def _assert_models_present() -> None:
    missing = [name for name in MODEL_FILES if not Path(MODELS_DIR, name).is_file()]
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
class Ideogram4:
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

        return create_app(COMFY_URL, CAPTION_TEMPLATE_PATH)

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
    headers. Serve it ephemerally, or set IDEOGRAM4_UI_REQUIRE_AUTH=1 and drive
    it from a client that can.
    """
    _launch_comfyui(UI_PORT, "0.0.0.0")


@app.local_entrypoint()
def main(
    prompt: str = "a vintage travel poster for the rings of Saturn, bold type reading 'SATURN'",
    output_dir: str = "outputs",
    preset: str = "Default",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    batch_size: int = 1,
) -> None:
    """End-to-end smoke test: `modal run app.py`."""
    params, images = Ideogram4().generate.remote(
        prompt=prompt,
        preset=preset,
        width=width,
        height=height,
        seed=None if seed < 0 else seed,
        batch_size=batch_size,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for index, data in enumerate(images):
        path = destination / f"ideogram4_{params['seed']}_{index}.png"
        path.write_bytes(data)
        print(f"wrote {path} ({len(data) / 1e6:.2f} MB)")
    print(f"seed={params['seed']} steps={params['steps']} {params['width']}x{params['height']}")
