"""RedCraft v3 (Krea 2) on Modal, served as a remote ComfyUI API.

RedCraft is a community finetune of Krea 2 turbo by `AiMetatron`, published on
Civitai under a listing that spans twenty versions across half a dozen unrelated
base models — Flux, SDXL, Z-Image, MiniMax H3, LTX. This deployment pins the
**"赤佬 3.0 (Krea2)"** version, which is the Krea 2 one.

Like the ultra, finepornv4 and redgpt2gpt services it carries no text encoder or
VAE of its own, so the Qwen3-VL-4B encoder and Qwen-Image VAE come from
Comfy-Org's Krea 2 mirror on Hugging Face — the same two files all four use.

    modal run app.py::download_models   # one-off, ~18 GB into a Volume
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

APP_NAME = "redcraft3-comfyui"

# --- Weights ----------------------------------------------------------------
# The finetune comes from Civitai with a verified digest; its companions are
# ungated on Hugging Face. Nothing here needs a token: unlike finepornv4 and
# redgpt2gpt, this listing is not NSFW-flagged and a ranged GET against the real
# download URL returns 206 anonymously. Verified, not assumed — if that ever
# changes, attaching a CIVITAI_TOKEN Secret to `download_models` is the whole
# fix, as those two services do.
MODEL_FILES = (
    # The file id carries the weight here, more than anywhere else in this repo:
    # this single version publishes fp8, int8, int4 and nvfp4 builds and gives
    # all four **the same filename**. The version id alone identifies none of
    # them; only the file id says which precision is fetched.
    weights.CivitaiFile(
        model_version_id=3139241,  # RedCraft 赤佬 3.0 (Krea2), published 2026-07-17
        file_id=3019490,  # the fp8 build, and the version's primary file
        # Renamed from redcraftHybridH3A2A_30Krea2.safetensors, which upstream
        # reuses verbatim for every precision.
        destination="diffusion_models/redcraft_v3_krea2_fp8.safetensors",
        sha256="f6088960c0febd27cbd372fc758bb07d012f2d8ae3cd10c45c903d48b94409ea",
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
    "redcraft3", ("diffusion_models", "text_encoders", "vae")
)

# ~18 GB of weights; the text encoder offloads after encoding, so the sampling
# working set is nearer 13 GB — the same shape as ultra and redgpt2gpt. L40S is
# the safe default. See README, "Choosing a GPU".
SETTINGS = service.Settings.from_env("REDCRAFT3", gpu="L40S")

models_volume = modal.Volume.from_name("redcraft3-models", create_if_missing=True)

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
class RedCraft3:
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
    prompt: str = "a rain-soaked neon alley at night, shot on a handheld camera",
    output_dir: str = "outputs",
    width: int = 1024,
    height: int = 1024,
    steps: int = 10,
    seed: int = -1,
    batch_size: int = 1,
) -> None:
    """End-to-end smoke test: `modal run app.py`."""
    params, images = RedCraft3().generate.remote(
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
        path = destination / f"redcraft3_{params['seed']}_{index}.png"
        path.write_bytes(data)
        print(f"wrote {path} ({len(data) / 1e6:.2f} MB)")
    print(
        f"seed={params['seed']} steps={params['steps']} cfg={params['cfg']} "
        f"{params['sampler_name']}/{params['scheduler']} "
        f"{params['width']}x{params['height']}"
    )
