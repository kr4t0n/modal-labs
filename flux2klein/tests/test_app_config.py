"""Checks on the Modal app definition that need no Modal credentials.

These target failures that otherwise surface minutes into a remote image build,
or worse, at model-load time inside a running container.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Both services ship top-level modules with these names, and pytest collects
# every suite in one interpreter. Drop whatever the other project's suite left
# in sys.modules so the imports below resolve against *this* project.
for _shared in ("workflow", "server", "app"):
    sys.modules.pop(_shared, None)

import app  # noqa: E402
import workflow  # noqa: E402


def test_extra_model_paths_is_valid_yaml_for_the_volume():
    config = yaml.safe_load(app.EXTRA_MODEL_PATHS_YAML)
    assert set(config) == {"flux2klein"}
    section = config["flux2klein"]
    assert section["base_path"] == app.MODELS_DIR
    # These keys are ComfyUI's own folder names. A typo yields an empty model
    # list and an error only once a prompt is queued.
    assert section["diffusion_models"] == "diffusion_models"
    assert section["text_encoders"] == "text_encoders"
    assert section["vae"] == "vae"


def test_downloaded_layout_matches_what_the_graph_asks_for():
    """Each file is renamed to an explicit destination; they must line up."""
    destinations = {m.destination for m in app.MODEL_FILES}
    for variant in workflow.VARIANTS.values():
        assert f"diffusion_models/{variant.checkpoint}" in destinations
        # Every variant's encoder must actually be downloaded, or the graph
        # validates locally and fails at queue time on the remote.
        assert f"text_encoders/{variant.text_encoder}" in destinations
    assert f"vae/{workflow.VAE}" in destinations


def test_every_downloaded_file_sits_under_a_configured_search_path():
    section = yaml.safe_load(app.EXTRA_MODEL_PATHS_YAML)["flux2klein"]
    configured = {value for key, value in section.items() if key != "base_path"}
    for model in app.MODEL_FILES:
        assert model.destination.split("/")[0] in configured, model.destination


def test_hf_secret_name_default():
    """`modal deploy` resolves this name; a wrong one fails the whole deploy."""
    assert app.HF_SECRET_NAME == "huggingface-secret"


def test_gated_repos_are_flagged():
    """The two transformers need an HF token; mislabelling them hides the 401."""
    gated = {m.repo_id for m in app.MODEL_FILES if m.gated}
    assert gated == {
        "black-forest-labs/FLUX.2-klein-base-9b-fp8",
        "black-forest-labs/FLUX.2-klein-9b-fp8",
        "ponpoke/flux2-klein-9b-uncensored-text-encoder",
    }
