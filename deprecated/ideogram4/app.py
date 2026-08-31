"""Ideogram 4 on Modal, served as a remote ComfyUI API.

Ideogram 4 is a 9.3B single-stream diffusion transformer with open weights and
first-class support in ComfyUI core. Rather than reimplement its sampling
schedule, this deployment runs a real headless ComfyUI on a Modal GPU and puts a
thin ASGI app in front of it. The resulting URL behaves like a ComfyUI server
(`/prompt`, `/history`, `/view`, `/ws`, the web UI) and additionally answers a
typed `/generate` contract that the bundled custom node calls.

Only the model-specific parts live here — the weight table, the graph, the
Modal object graph. The container image, the ComfyUI supervisor and the ASGI
layer come from `comfyui_modal`.

    modal run app.py::download_models   # one-off, ~30 GB into a Volume
    modal deploy app.py                 # the API
    modal serve app.py                  # the browser UI, ephemeral

Deploy-time configuration comes from the environment; see .env.example.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
# `add_local_python_source` resolves modules through the local interpreter, so
# both the sibling modules and the shared package at the repository root have to
# be importable no matter which directory modal is invoked from.
# Retired: one level deeper than a live service, so the repository root
# is two parents up rather than one.
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from comfyui_modal import service  # noqa: E402
from comfyui_modal.service import (  # noqa: E402
    COMFY_HOST,
    COMFY_PORT,
    COMFY_URL,
    MODELS_DIR,
    UI_PORT,
)

APP_NAME = "ideogram4-comfyui"
CAPTION_TEMPLATE_PATH = "/root/assets/magic_prompt_template.txt"

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

EXTRA_MODEL_PATHS_YAML = service.extra_model_paths_yaml(
    "ideogram4", ("diffusion_models", "text_encoders", "vae")
)

# 29.5 GB of fp8 weights total; ~18.9 GB is resident during sampling, once the
# text encoder is offloaded. A 48 GB L40S therefore holds the lot and is roughly
# half H100's hourly rate — but it is also slower, so compare cost per image
# rather than per hour before switching. See README, "Choosing a GPU".
SETTINGS = service.Settings.from_env("IDEOGRAM4", gpu="H100")

models_volume = modal.Volume.from_name("ideogram4-models", create_if_missing=True)

image = service.build_image(["workflow", "server", "comfyui_modal"]).add_local_file(
    HERE / "assets" / "magic_prompt_template.txt", CAPTION_TEMPLATE_PATH
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
class Ideogram4:
    """A container running ComfyUI, fronted by the ASGI app in server.py."""

    @modal.enter()
    def start_comfyui(self) -> None:
        self.process = service.launch_comfyui(
            COMFY_PORT,
            COMFY_HOST,
            required_models=MODEL_FILES,
            extra_paths_yaml=EXTRA_MODEL_PATHS_YAML,
        )
        service.wait_for_comfyui(self.process)

    @modal.exit()
    def stop_comfyui(self) -> None:
        service.stop_comfyui(self.process)

    @modal.asgi_app(requires_proxy_auth=SETTINGS.require_auth)
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
    headers. Serve it ephemerally, or set IDEOGRAM4_UI_REQUIRE_AUTH=1 and drive
    it from a client that can.
    """
    service.launch_comfyui(
        UI_PORT,
        "0.0.0.0",
        required_models=MODEL_FILES,
        extra_paths_yaml=EXTRA_MODEL_PATHS_YAML,
    )


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
