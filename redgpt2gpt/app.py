"""RedGPT2 (Krea 2) on Modal, served as a remote ComfyUI API.

RedGPT2 is a community finetune of Krea 2 turbo by `AiMetatron`, published on
Civitai under a listing that carries several editions. This deployment pins the
**"KREA2 GPT 逼真版"** (GPT photorealistic) edition — a single ~12.8 GB fp8
diffusion model.

That distinction matters: the listing is titled "Alternating Evaluation", and a
*different* edition on the same page ships two checkpoints (high-noise and
low-noise) sampled alternately. This one does not. See AGENTS.md.

Like the ultra and finepornv4 services it carries no text encoder or VAE of its
own, so the Qwen3-VL-4B encoder and Qwen-Image VAE come from Comfy-Org's Krea 2
mirror on Hugging Face — the same two files all three use.

    modal secret create civitai-secret CIVITAI_TOKEN=...
    modal run app.py::download_models   # one-off, ~18 GB into a Volume
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

APP_NAME = "redgpt2gpt-comfyui"

# --- Weights ----------------------------------------------------------------
# The finetune comes from Civitai with a verified digest; its companions are
# ungated on Hugging Face.
#
# The download is authenticated: this model is NSFW-flagged and Civitai answers
# 401 to an unauthenticated request for it — verified against the real URL, not
# assumed. Only `download_models` carries the secret; the serving containers
# read the Volume and need nothing.
CIVITAI_SECRET_NAME = os.environ.get("REDGPT2GPT_CIVITAI_SECRET", "civitai-secret")

MODEL_FILES = (
    # Pinned by version *and* file id. The listing carries several editions and
    # more than one of them reuses a filename, so the version alone would not
    # identify this build.
    weights.CivitaiFile(
        model_version_id=3123514,  # KREA2 GPT photorealistic edition, published 2026-07-13
        file_id=3004003,
        # Renamed from redgpt2Krea2Turbo_krea2GPT.safetensors, which does not
        # say which edition or precision it is.
        destination="diffusion_models/redgpt2_krea2_gpt_fp8.safetensors",
        sha256="acd064df6b24457abb13de9ce28917d5c4269c33384f8c16f7a1749c08dd33da",
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
    "redgpt2gpt", ("diffusion_models", "text_encoders", "vae")
)

# ~18 GB of weights; the text encoder offloads after encoding, so the sampling
# working set is nearer 13 GB — the same shape as the ultra service, and much
# lighter than finepornv4's bf16 build. L40S is the safe default. See README,
# "Choosing a GPU".
SETTINGS = service.Settings.from_env("REDGPT2GPT", gpu="L40S")

models_volume = modal.Volume.from_name("redgpt2gpt-models", create_if_missing=True)

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
class RedGPT2GPT:
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
    prompt: str = "a portrait of a woman reading by a window in late afternoon light",
    output_dir: str = "outputs",
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: int = -1,
    batch_size: int = 1,
) -> None:
    """End-to-end smoke test: `modal run app.py`."""
    params, images = RedGPT2GPT().generate.remote(
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
        path = destination / f"redgpt2gpt_{params['seed']}_{index}.png"
        path.write_bytes(data)
        print(f"wrote {path} ({len(data) / 1e6:.2f} MB)")
    print(
        f"seed={params['seed']} steps={params['steps']} cfg={params['cfg']} "
        f"{params['sampler_name']}/{params['scheduler']} "
        f"{params['width']}x{params['height']}"
    )
