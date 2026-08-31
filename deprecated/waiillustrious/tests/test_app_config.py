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


def test_extra_model_paths_points_at_the_checkpoints_folder():
    config = yaml.safe_load(app.EXTRA_MODEL_PATHS_YAML)
    assert set(config) == {"waiillustrious"}
    section = config["waiillustrious"]
    assert section["base_path"] == app.MODELS_DIR
    # ComfyUI's own folder name; a typo yields an empty checkpoint list and an
    # error only once a prompt is queued.
    assert section["checkpoints"] == "checkpoints"


def test_download_destination_matches_what_the_graph_asks_for():
    assert app.CHECKPOINT_FILE.destination == f"checkpoints/{workflow.CHECKPOINT}"
    assert (app.CHECKPOINT_FILE.destination,) == app.REQUIRED_MODELS
    assert app.CHECKPOINT_FILE.destination.split("/")[0] == "checkpoints"


def test_checksum_is_a_well_formed_sha256():
    """The digest is the only integrity check on a third-party CDN download."""
    assert re.fullmatch(r"[0-9a-f]{64}", app.CHECKPOINT_FILE.sha256)


def test_download_url_pins_the_model_version():
    """A floating 'latest' URL would silently change the model under us."""
    url = app.CHECKPOINT_FILE.url
    assert str(app.CHECKPOINT_FILE.model_version_id) in url
    assert str(app.CHECKPOINT_FILE.file_id) in url
    assert url.startswith("https://civitai.com/api/download/models/")


def test_gpu_default_is_the_cheap_card():
    """~6.8 GB fp16 fits a 24 GB A10; defaulting higher just costs money."""
    assert app.SETTINGS.gpu == "A10"
