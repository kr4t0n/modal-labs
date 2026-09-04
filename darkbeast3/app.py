"""Dark Beast v3 (Krea 2) on Modal, served as a remote ComfyUI API.

Dark Beast is a community finetune of Krea 2 turbo by `AiMetatron`, published on
Civitai under a listing titled "H3 Director Edition" that spans fifteen versions
across MiniMax H3, Z-Image, FLUX.2 klein, SDXL and Krea 2. This deployment pins
**"Dark Beast 3 黑兽3.0"**, which is the Krea 2 one.

The H3 in that title is a *video* model and a different version on the same page
(`3274224`). Its short-film pipeline, 2K upscaling and 6-10 step guidance do not
apply here. See AGENTS.md.

Like the ultra, finepornv4, redgpt2gpt and redcraft3 services it carries no
text encoder or VAE of its own, so the Qwen3-VL-4B encoder and Qwen-Image VAE
come from Comfy-Org's Krea 2 mirror on Hugging Face — the same two files all
five use.

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

APP_NAME = "darkbeast3-comfyui"

# --- Weights ----------------------------------------------------------------
# The finetune comes from Civitai with a verified digest; its companions are
# ungated on Hugging Face. Nothing here needs a token.
#
# Worth stating precisely, because the obvious heuristic is wrong: this listing
# *is* NSFW-flagged, and it still serves an anonymous ranged GET (206, real
# safetensors bytes, checked against the live URL). The flag does not decide
# gating — finepornv4 and redgpt2gpt are flagged and 401. Each service records
# what was observed rather than inferred.
MODEL_FILES = (
    # The file id is what selects the precision: this one version publishes
    # int8, fp8, bf16, nvfp4 and int4 builds and gives all five the same
    # filename. The version id identifies none of them.
    weights.CivitaiFile(
        model_version_id=3173268,  # Dark Beast 3 黑兽3.0 (Krea 2), published 2026-07-28
        file_id=3053854,  # the int8 build, and the version's primary file
        # Renamed from darkBeastH3Director_darkBeast330.safetensors, which names
        # neither the precision nor which base model it is.
        destination="diffusion_models/darkbeast_v3_krea2_int8.safetensors",
        sha256="b60cb86fc1c8a84f37991c0c4d9bffe9ba4a2ce6ae1ec26ff2691bb21d87c433",
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
    "darkbeast3", ("diffusion_models", "text_encoders", "vae")
)

# ~19 GB of weights; the text encoder offloads after encoding, so the sampling
# working set is nearer 14 GB — the same shape as ultra, whose checkpoint is
# also a 13.8 GB int8 build. L40S is the safe default. See README.
SETTINGS = service.Settings.from_env("DARKBEAST3", gpu="L40S")

models_volume = modal.Volume.from_name("darkbeast3-models", create_if_missing=True)

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
class DarkBeast3:
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
    prompt: str = "a close-up portrait in hard directional light, heavy film grain",
    output_dir: str = "outputs",
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    seed: int = -1,
    batch_size: int = 1,
) -> None:
    """End-to-end smoke test: `modal run app.py`."""
    params, images = DarkBeast3().generate.remote(
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
        path = destination / f"darkbeast3_{params['seed']}_{index}.png"
        path.write_bytes(data)
        print(f"wrote {path} ({len(data) / 1e6:.2f} MB)")
    print(
        f"seed={params['seed']} steps={params['steps']} cfg={params['cfg']} "
        f"{params['sampler_name']}/{params['scheduler']} "
        f"{params['width']}x{params['height']}"
    )
