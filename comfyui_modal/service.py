"""Modal-side plumbing shared by every ComfyUI service.

The Modal decorators themselves have to live in each service's `app.py` — Modal
discovers them at module scope in the file you deploy — but everything they call
is here: the container image, the deploy-time settings, and the supervisor that
starts ComfyUI and waits for it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import modal

# Ideogram4Scheduler, DualModelGuider, CFGOverride and Flux2Scheduler are all
# ComfyUI core nodes; this is the release every service's graph is built against.
COMFYUI_REF = "v0.34.2"
COMFYUI_DIR = "/root/ComfyUI"
MODELS_DIR = "/root/models"

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
UI_PORT = 8000


def env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", ""}


def single_line(*commands: str) -> tuple[str, ...]:
    """Reject multi-line build commands at import time.

    Each `run_commands` entry becomes one Dockerfile RUN line. Interpolating a
    multi-line value into one produces a Dockerfile that fails to parse, and the
    error surfaces minutes into a remote build rather than here.
    """
    for command in commands:
        if "\n" in command:
            raise ValueError(f"run_commands entry spans multiple lines: {command!r}")
    return commands


@dataclass(frozen=True)
class Settings:
    """Deploy-time configuration, read from `<PREFIX>_*` environment variables.

    Modal evaluates the app file when you deploy, so these are baked into the
    deployment; changing one means re-deploying.
    """

    gpu: str
    min_containers: int
    max_containers: int
    scaledown_window: int
    concurrent_inputs: int
    require_auth: bool
    ui_require_auth: bool

    @classmethod
    def from_env(cls, prefix: str, *, gpu: str = "H100") -> Settings:
        def value(name: str, default: str) -> str:
            return os.environ.get(f"{prefix}_{name}", default)

        return cls(
            gpu=value("GPU", gpu),
            min_containers=int(value("MIN_CONTAINERS", "0")),
            # Defaults to one so the ComfyUI queue, /history and /view stay
            # coherent: those are per-container state, and a second replica
            # would answer for prompts it never ran.
            max_containers=int(value("MAX_CONTAINERS", "1")),
            scaledown_window=int(value("SCALEDOWN_WINDOW", "300")),
            # Concurrency is about not blocking the proxy: ComfyUI serialises
            # sampling itself, but progress polls and /view fetches have to get
            # through mid-render.
            concurrent_inputs=int(value("CONCURRENT_INPUTS", "20")),
            require_auth=env_flag(f"{prefix}_REQUIRE_AUTH", "1"),
            # Off by default because browsers cannot attach proxy-auth headers;
            # that endpoint is meant for `modal serve`, not `modal deploy`.
            ui_require_auth=env_flag(f"{prefix}_UI_REQUIRE_AUTH", "0"),
        )


def extra_model_paths_yaml(section: str, folders: Iterable[str]) -> str:
    """A ComfyUI `extra_model_paths.yaml` pointing at the weights Volume.

    Search paths rather than a bind-mount over ComfyUI/models, which would hide
    the config files the repo ships there.
    """
    lines = [f"{section}:", f"  base_path: {MODELS_DIR}"]
    lines += [f"  {folder}: {folder}" for folder in folders]
    return "\n".join(lines) + "\n"


def build_image(python_sources: Sequence[str], *, comfyui_ref: str = COMFYUI_REF) -> modal.Image:
    """The container image: torch, a pinned ComfyUI checkout, and the ASGI deps."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install("torch==2.13.0", "torchvision==0.28.0", "torchaudio==2.11.0")
        .run_commands(
            *single_line(
                f"git clone --depth 1 --branch {comfyui_ref}"
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
        .add_local_python_source(*python_sources)
    )


def assert_models_present(required: Sequence[str]) -> None:
    missing = [name for name in required if not Path(MODELS_DIR, name).is_file()]
    if missing:
        raise RuntimeError(
            "weights Volume is missing "
            + ", ".join(missing)
            + " — run `modal run app.py::download_models` first"
        )


def launch_comfyui(
    port: int, listen: str, *, required_models: Sequence[str], extra_paths_yaml: str
) -> subprocess.Popen:
    """Start ComfyUI, after checking the weights are actually there."""
    assert_models_present(required_models)
    # ComfyUI loads this from its own directory during startup. Written here
    # rather than baked into the image: `run_commands` entries become Dockerfile
    # RUN lines, and a multi-line document in one is a parse error.
    Path(COMFYUI_DIR, "extra_model_paths.yaml").write_text(extra_paths_yaml)
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


def wait_for_comfyui(process: subprocess.Popen, url: str = COMFY_URL, timeout: float = 600.0):
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


def stop_comfyui(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
