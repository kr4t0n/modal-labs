"""FLUX.2 [klein] 9B on Modal, served as a remote ComfyUI API.

FLUX.2 klein is Black Forest Labs' compact FLUX.2: a 9B transformer with a
Qwen3-8B text encoder, shipped in two flavours — an undistilled `base` that
responds to CFG and negative prompts, and a 4-step guidance-`distilled` build.
Both are supported by ComfyUI core.

Only the model-specific parts live here — the weight table, the graph, the Modal
object graph. The container image, the ComfyUI supervisor and the ASGI layer come
from `comfyui_modal`.

    modal secret create huggingface-secret HF_TOKEN=hf_...   # transformers are gated
    modal run app.py::download_models                 # one-off, ~44 GB
    modal deploy app.py                               # the API
    modal serve app.py                                # the browser UI, ephemeral

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

APP_NAME = "flux2klein-comfyui"

# --- Weights ----------------------------------------------------------------
# Unlike the ideogram4 service these come from four repos whose internal layouts
# do not match ComfyUI's, so every file carries an explicit destination.
HF_SECRET_NAME = os.environ.get("FLUX2KLEIN_HF_SECRET", "huggingface-secret")


MODEL_FILES = (
    weights.HuggingFaceFile(
        "black-forest-labs/FLUX.2-klein-base-9b-fp8",
        "flux-2-klein-base-9b-fp8.safetensors",
        "diffusion_models/flux-2-klein-base-9b-fp8.safetensors",
        gated=True,
    ),
    weights.HuggingFaceFile(
        "black-forest-labs/FLUX.2-klein-9b-fp8",
        "flux-2-klein-9b-fp8.safetensors",
        "diffusion_models/flux-2-klein-9b-fp8.safetensors",
        gated=True,
    ),
    weights.HuggingFaceFile(
        "Comfy-Org/flux2-klein-9B",
        "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
        "text_encoders/qwen_3_8b_fp8mixed.safetensors",
        gated=False,
    ),
    weights.HuggingFaceFile(
        "ponpoke/flux2-klein-9b-uncensored-text-encoder",
        "model.safetensors",
        "text_encoders/qwen_3_8b_uncensored_bf16.safetensors",
        gated=True,
    ),
    weights.HuggingFaceFile(
        "black-forest-labs/FLUX.2-small-decoder",
        "full_encoder_small_decoder.safetensors",
        "vae/full_encoder_small_decoder.safetensors",
        gated=False,
    ),
    # Adapters. Small enough that fetching them unconditionally costs little,
    # and a request naming one must not have to wait for a download.
    weights.HuggingFaceFile(
        "Ashen3/SNOFS",
        "Klein9b/klein_snofs_v1_4.safetensors",
        "loras/klein_snofs_v1_4.safetensors",
        gated=False,
    ),
)

REQUIRED_MODELS = weights.destinations(MODEL_FILES)

EXTRA_MODEL_PATHS_YAML = service.extra_model_paths_yaml(
    "flux2klein", ("diffusion_models", "text_encoders", "vae", "loras")
)

# One transformer and one encoder are resident at a time: ~18 GB for base and
# distilled, ~26 GB for ponpoke-uncensored (bf16 encoder). Still smaller than the
# ideogram4 service. Compare cost per image, not per hour — see README.
SETTINGS = service.Settings.from_env("FLUX2KLEIN", gpu="H100")

models_volume = modal.Volume.from_name("flux2klein-models", create_if_missing=True)

image = service.build_image(["workflow", "server", "comfyui_modal"])

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={MODELS_DIR: models_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])],
)
def download_models(force: bool = False) -> list[str]:
    """Populate the weights Volume.

    The two transformers and the uncensored encoder are gated, so the Hugging
    Face account behind HF_TOKEN must have accepted each licence first.
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
class Flux2Klein:
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
    headers. Serve it ephemerally, or set FLUX2KLEIN_UI_REQUIRE_AUTH=1 and drive
    it from a client that can.
    """
    service.launch_comfyui(
        UI_PORT,
        "0.0.0.0",
        required_models=REQUIRED_MODELS,
        extra_paths_yaml=EXTRA_MODEL_PATHS_YAML,
    )


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
