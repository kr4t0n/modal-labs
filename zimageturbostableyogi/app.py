"""Z-Image Turbo (Stable Yogi) on Modal, served as a remote ComfyUI API.

Stable Yogi's finetune of Alibaba's Z-Image Turbo, distributed on Civitai as a
6 GB fp8 diffusion model. It carries no text encoder or VAE of its own, so the
Qwen3-4B encoder and autoencoder are fetched from Comfy-Org's Z-Image mirror.

Unlike the other Civitai-sourced service here, **this one requires a token**:
every version 401s anonymously. The digest is still pinned, so a substituted
file fails closed regardless.

    modal secret create civitai-secret CIVITAI_TOKEN=...   # required
    modal run app.py::download_models                      # one-off, ~12 GB
    modal deploy app.py                                    # the API
    modal serve app.py                                     # the browser UI

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

APP_NAME = "zimageturbostableyogi-comfyui"

# Every version of this model 401s without a token, so the secret is not
# optional the way it is for the other Civitai-sourced service.
CIVITAI_SECRET_NAME = os.environ.get("ZIMAGETURBOSTABLEYOGI_CIVITAI_SECRET", "civitai-secret")

# --- Weights ----------------------------------------------------------------
# The fp8 build rather than the newer NVFP4 one: NVFP4 needs Blackwell, which is
# far more card than a 6 GB turbo model warrants. The int8-convrot build is the
# fallback for Ampere, which has int8 but no fp8.
MODEL_FILES = (
    weights.CivitaiFile(
        model_version_id=3096324,  # "2603 Fp8", published 2026-07-04
        file_id=2975960,
        destination="diffusion_models/zimageturbostableyogi.safetensors",
        sha256="8be4161e7d6ec8a6c714e4c62a4856b92e59d16102b7d7927f0964bbf3a5fa32",
    ),
    weights.HuggingFaceFile(
        "Comfy-Org/z_image_turbo",
        "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
        "text_encoders/qwen_3_4b_fp8_mixed.safetensors",
    ),
    weights.HuggingFaceFile(
        "Comfy-Org/z_image_turbo",
        "split_files/vae/ae.safetensors",
        "vae/ae.safetensors",
    ),
)

REQUIRED_MODELS = weights.destinations(MODEL_FILES)

EXTRA_MODEL_PATHS_YAML = service.extra_model_paths_yaml(
    "zimageturbostableyogi", ("diffusion_models", "text_encoders", "vae")
)

# ~12 GB of weights; the text encoder offloads after encoding, so the sampling
# working set is nearer 7 GB. L4 is the cheapest Modal card with fp8 tensor
# cores (Ada, capability 8.9), which the F8_E4M3 weights need, and 8-step turbo
# sampling is a light enough workload to suit it. See README, "Choosing a GPU".
SETTINGS = service.Settings.from_env("ZIMAGETURBOSTABLEYOGI", gpu="L4")

models_volume = modal.Volume.from_name("zimageturbostableyogi-models", create_if_missing=True)

image = service.build_image(["workflow", "server", "comfyui_modal"])

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={MODELS_DIR: models_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name(CIVITAI_SECRET_NAME, required_keys=["CIVITAI_TOKEN"])],
)
def download_models(force: bool = False) -> list[str]:
    """Populate the weights Volume, verifying the Civitai digest.

    The checkpoint needs CIVITAI_TOKEN; its companions on Hugging Face do not.
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
class ZImageTurboStableYogi:
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
    prompt: str = "a harbour at dawn, fishing boats and pastel houses, soft light",
    output_dir: str = "outputs",
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: int = -1,
    batch_size: int = 1,
) -> None:
    """End-to-end smoke test: `modal run app.py`."""
    params, images = ZImageTurboStableYogi().generate.remote(
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
        path = destination / f"zimage_{params['seed']}_{index}.png"
        path.write_bytes(data)
        print(f"wrote {path} ({len(data) / 1e6:.2f} MB)")
    print(
        f"seed={params['seed']} steps={params['steps']} cfg={params['cfg']} "
        f"shift={params['shift']} {params['width']}x{params['height']}"
    )
