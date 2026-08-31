"""ULTRA (Krea 2) on Modal, served as a remote ComfyUI API.

ULTRA is a community finetune of Krea 2 by `AIA_civit`, distributed on Civitai as
a ~13.8 GB int8 diffusion model. It carries no text encoder or VAE of its own, so
the Qwen3-VL-4B encoder and Qwen-Image VAE are fetched from Comfy-Org's Krea 2
mirror on Hugging Face.

The checkpoint is pulled from **civitai.com** and verified against the digest
Civitai publishes. That matters here: the model was originally pointed at via a
lookalike mirror domain, and a pinned digest makes a substituted file fail closed
regardless of which host serves the bytes.

    modal run app.py::download_models   # one-off, ~19 GB into a Volume
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
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from comfyui_modal import service, weights  # noqa: E402
from comfyui_modal.service import (  # noqa: E402
    COMFY_HOST,
    COMFY_PORT,
    COMFY_URL,
    MODELS_DIR,
    UI_PORT,
)

APP_NAME = "ultra-comfyui"

# --- Weights ----------------------------------------------------------------
# The finetune comes from Civitai with a verified digest; its companions are
# ungated on Hugging Face. Nothing here needs a token.
MODEL_FILES = (
    weights.CivitaiFile(
        model_version_id=3215898,  # ULTRA v15, published 2026-08-29
        file_id=3139476,
        destination="diffusion_models/ultra_v15.safetensors",
        sha256="43d561de50adb20c459967457fd64600286a3639571f5fdd6b1b067e51b2f576",
    ),
    weights.HuggingFaceFile(
        "Comfy-Org/Krea-2",
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
    ),
    weights.HuggingFaceFile(
        "Comfy-Org/Krea-2",
        "vae/qwen_image_vae.safetensors",
        "vae/qwen_image_vae.safetensors",
    ),
)

REQUIRED_MODELS = weights.destinations(MODEL_FILES)

EXTRA_MODEL_PATHS_YAML = service.extra_model_paths_yaml(
    "ultra", ("diffusion_models", "text_encoders", "vae")
)

# ~19 GB of weights; ComfyUI offloads the text encoder after encoding, so the
# sampling working set is nearer 14 GB. L40S is the safe default — a 24 GB A10
# very likely fits and is cheaper, but would be tight if both stayed resident.
# See README, "Choosing a GPU".
SETTINGS = service.Settings.from_env("ULTRA", gpu="L40S")

models_volume = modal.Volume.from_name("ultra-models", create_if_missing=True)

image = service.build_image(["workflow", "server", "comfyui_modal"])

app = modal.App(APP_NAME, image=image)


@app.function(volumes={MODELS_DIR: models_volume}, timeout=7200)
def download_models(force: bool = False) -> list[str]:
    """Populate the weights Volume, verifying the Civitai digest."""
    written = weights.download_weights(MODEL_FILES, MODELS_DIR, force=force)
    models_volume.commit()
    return written


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
class Ultra:
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
    """The ComfyUI web interface, for `modal serve app.py`."""
    service.launch_comfyui(
        UI_PORT,
        "0.0.0.0",
        required_models=REQUIRED_MODELS,
        extra_paths_yaml=EXTRA_MODEL_PATHS_YAML,
    )


@app.local_entrypoint()
def main(
    prompt: str = "a low angle action shot of a cyclist on a rain-slick street at dusk",
    output_dir: str = "outputs",
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: int = -1,
    batch_size: int = 1,
) -> None:
    """End-to-end smoke test: `modal run app.py`."""
    params, images = Ultra().generate.remote(
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
        path = destination / f"ultra_{params['seed']}_{index}.png"
        path.write_bytes(data)
        print(f"wrote {path} ({len(data) / 1e6:.2f} MB)")
    print(
        f"seed={params['seed']} steps={params['steps']} cfg={params['cfg']} "
        f"{params['width']}x{params['height']}"
    )
