"""Checks on the Modal app definition that need no Modal credentials."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _shared in ("workflow", "server", "app"):
    sys.modules.pop(_shared, None)

import app  # noqa: E402
import workflow  # noqa: E402
from comfyui_modal import weights  # noqa: E402


def test_extra_model_paths_covers_every_folder_used():
    config = yaml.safe_load(app.EXTRA_MODEL_PATHS_YAML)
    assert set(config) == {"zimageturbostableyogi"}
    section = config["zimageturbostableyogi"]
    assert section["base_path"] == app.MODELS_DIR
    configured = {v for k, v in section.items() if k != "base_path"}
    for destination in app.REQUIRED_MODELS:
        assert destination.split("/")[0] in configured, destination


def test_downloaded_layout_matches_what_the_graph_asks_for():
    destinations = set(app.REQUIRED_MODELS)
    assert f"diffusion_models/{workflow.DIFFUSION_MODEL}" in destinations
    assert f"text_encoders/{workflow.TEXT_ENCODER}" in destinations
    assert f"vae/{workflow.VAE}" in destinations


def test_the_checkpoint_comes_from_civitai_with_a_pinned_digest():
    """The finetune is the one file with no upstream integrity guarantee."""
    checkpoint = next(f for f in app.MODEL_FILES if isinstance(f, weights.CivitaiFile))
    assert re.fullmatch(r"[0-9a-f]{64}", checkpoint.sha256)
    # A floating "latest" URL would silently change the model under us.
    assert str(checkpoint.model_version_id) in checkpoint.url
    assert checkpoint.url.startswith("https://civitai.com/api/download/models/")


def test_companions_come_from_an_ungated_hugging_face_mirror():
    """Only the checkpoint needs a token; its companions must not."""
    hf = [f for f in app.MODEL_FILES if isinstance(f, weights.HuggingFaceFile)]
    assert len(hf) == 2
    assert all(not f.gated for f in hf)
    assert all(f.repo_id == "Comfy-Org/z_image_turbo" for f in hf)


def test_the_civitai_secret_is_required():
    """Every version of this model 401s anonymously, unlike the ultra service."""
    assert app.CIVITAI_SECRET_NAME == "civitai-secret"
