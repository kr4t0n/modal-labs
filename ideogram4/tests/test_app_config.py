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
    assert set(config) == {"ideogram4"}
    section = config["ideogram4"]
    assert section["base_path"] == app.MODELS_DIR
    # These keys are ComfyUI's own folder names. A typo yields an empty model
    # list and an error only once a prompt is queued.
    assert section["diffusion_models"] == "diffusion_models"
    assert section["text_encoders"] == "text_encoders"
    assert section["vae"] == "vae"


def test_downloaded_layout_matches_what_the_graph_asks_for():
    """hf_hub_download preserves repo subpaths, so the two must line up."""
    downloaded = set(app.MODEL_FILES)
    assert f"diffusion_models/{workflow.DIFFUSION_MODEL}" in downloaded
    assert f"diffusion_models/{workflow.UNCONDITIONAL_DIFFUSION_MODEL}" in downloaded
    assert f"text_encoders/{workflow.TEXT_ENCODER}" in downloaded
    assert f"vae/{workflow.VAE}" in downloaded


def test_every_downloaded_file_sits_under_a_configured_search_path():
    section = yaml.safe_load(app.EXTRA_MODEL_PATHS_YAML)["ideogram4"]
    configured = {value for key, value in section.items() if key != "base_path"}
    for filename in app.MODEL_FILES:
        assert filename.split("/")[0] in configured, filename
