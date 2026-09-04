"""FinePorn v4 (Krea 2) on Modal, served as a remote ComfyUI API.

FinePorn is a community merge of Krea 2 turbo by `Reevo`, distributed on Civitai
in four precisions. This deployment pins the **bf16** build — ~25.7 GB, the
largest and slowest of the four, and the one its author describes as the most
accurate. See README, "Choosing a precision", for the cheaper alternatives.

Like the ULTRA service it carries no text encoder or VAE of its own, so the
Qwen3-VL-4B encoder and Qwen-Image VAE come from Comfy-Org's Krea 2 mirror on
Hugging Face — the same two files, byte for byte.

The checkpoint is pulled from **civitai.com** and verified against the digest
Civitai publishes, so a substituted file fails closed regardless of which host
serves the bytes.

    modal run app.py::download_models   # one-off, ~30 GB into a Volume
    modal deploy app.py                 # the API
    modal serve app.py                  # the browser UI, ephemeral

Deploy-time configuration comes from the environment; see .env.example.
"""

from __future__ import annotations

import os
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

APP_NAME = "finepornv4-comfyui"

# --- Weights ----------------------------------------------------------------
# The merge comes from Civitai with a verified digest; its companions are
# ungated on Hugging Face.
#
# Unlike the ultra service, this download is *not* anonymous: the model is
# NSFW-flagged, and Civitai answers 401 to an unauthenticated request for it.
# Only `download_models` carries the secret — the serving containers read the
# Volume and need nothing.
CIVITAI_SECRET_NAME = os.environ.get("FINEPORNV4_CIVITAI_SECRET", "civitai-secret")

MODEL_FILES = (
    # The version id is load-bearing: this model publishes int8, nvfp4, fp8 and
    # bf16 builds, and several of them share a filename upstream. Pinning the
    # version *and* the file id is what makes "the bf16 one" unambiguous.
    weights.CivitaiFile(
        model_version_id=3197873,  # FinePorn v4 bf16, published 2026-08-04
        file_id=3079078,
        # Renamed from finepornV4INT8NVFP4BF16_v4Bf16.safetensors, whose name
        # lists all four precisions for a file that is only one of them.
        destination="diffusion_models/fineporn_v4_bf16.safetensors",
        sha256="532b1cbcaf09c478a91c01928a63f6b16dcff5585d9be5191c2c8973a16cda5e",
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
    "finepornv4", ("diffusion_models", "text_encoders", "vae")
)

# ~30 GB of weights. ComfyUI offloads the text encoder after encoding, so the
# sampling working set is the 25.7 GB transformer plus activations — and this
# service defaults to 1280x1280, which makes those activations larger than the
# 1 MP the other services assume. H100 is the safe default; a 48 GB L40S fits
# and is cheaper. See README, "Choosing a GPU".
SETTINGS = service.Settings.from_env("FINEPORNV4", gpu="H100")

models_volume = modal.Volume.from_name("finepornv4-models", create_if_missing=True)

image = service.build_image(["workflow", "server", "comfyui_modal"])

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={MODELS_DIR: models_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name(CIVITAI_SECRET_NAME, required_keys=["CIVITAI_TOKEN"])],
)
def download_models(force: bool = False) -> list[str]:
    """Populate the weights Volume, verifying the Civitai digest.

    Needs a Civitai API token; see README, "Setup".
    """
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
class FinePornV4:
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
    prompt: str = (
        "this is an amateur photo taken from smartphone, casual photo of a "
        "woman laughing in a sunlit kitchen"
    ),
    output_dir: str = "outputs",
    width: int = 1280,
    height: int = 1280,
    steps: int = 10,
    seed: int = -1,
    batch_size: int = 1,
) -> None:
    """End-to-end smoke test: `modal run app.py`."""
    params, images = FinePornV4().generate.remote(
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
        path = destination / f"finepornv4_{params['seed']}_{index}.png"
        path.write_bytes(data)
        print(f"wrote {path} ({len(data) / 1e6:.2f} MB)")
    print(
        f"seed={params['seed']} steps={params['steps']} cfg={params['cfg']} "
        f"{params['sampler_name']}/{params['scheduler']} "
        f"{params['width']}x{params['height']}"
    )
