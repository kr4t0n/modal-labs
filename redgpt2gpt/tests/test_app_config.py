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
    assert set(config) == {"redgpt2gpt"}
    section = config["redgpt2gpt"]
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


def test_the_pinned_edition_is_the_single_file_gpt_one():
    """The listing carries several editions, including a two-file AE build.

    Pinning the version alone would not be enough — editions on that page reuse
    filenames — so the file id is pinned too and asserted here.
    """
    checkpoint = next(f for f in app.MODEL_FILES if isinstance(f, weights.CivitaiFile))
    assert checkpoint.model_version_id == 3123514
    assert checkpoint.file_id == 3004003
    assert f"fileId={checkpoint.file_id}" in checkpoint.url
    # Exactly one checkpoint: the alternating-evaluation edition would need two.
    assert sum(isinstance(f, weights.CivitaiFile) for f in app.MODEL_FILES) == 1


def test_the_civitai_secret_is_wired():
    """NSFW-flagged upstream, so Civitai 401s an unauthenticated download.

    Same shape as flux2klein's `test_both_secrets_are_wired_for_the_two_weight_sources`;
    only Civitai here, since the Hugging Face companions are ungated.

    Without the token Civitai may answer 200 with an HTML error page rather than
    failing outright, so dropping this would surface as a checksum mismatch
    rather than an auth error.
    """
    assert app.CIVITAI_SECRET_NAME == "civitai-secret"


def test_companions_come_from_an_ungated_hugging_face_mirror():
    """Nothing here should need a token; a gated flag would break deploys."""
    hf = [f for f in app.MODEL_FILES if isinstance(f, weights.HuggingFaceFile)]
    assert len(hf) == 2
    assert all(not f.gated for f in hf)
    assert all(f.repo_id == "Comfy-Org/Krea-2" for f in hf)


def test_the_krea2_companions_match_the_ultra_service_exactly():
    """Three services, one base model, one pair of companion files.

    finepornv4 has the same assertion against ultra, so pinning both to ultra
    transitively pins all three. If these diverge it means one picked up a
    different encoder or VAE build, which changes output without failing.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ultra"))
    for name in ("workflow", "server", "app"):
        sys.modules.pop(name, None)
    try:
        import app as ultra_app
        import workflow as ultra_workflow

        assert ultra_workflow.TEXT_ENCODER == workflow.TEXT_ENCODER
        assert ultra_workflow.VAE == workflow.VAE
        assert ultra_workflow.CLIP_TYPE == workflow.CLIP_TYPE

        ultra_hf = {
            (f.repo_id, f.filename)
            for f in ultra_app.MODEL_FILES
            if isinstance(f, weights.HuggingFaceFile)
        }
        ours = {
            (f.repo_id, f.filename)
            for f in app.MODEL_FILES
            if isinstance(f, weights.HuggingFaceFile)
        }
        assert ultra_hf == ours
    finally:
        # Put this project's modules back for whatever runs next in this file.
        sys.path.pop(0)
        for name in ("workflow", "server", "app"):
            sys.modules.pop(name, None)


def test_this_service_does_not_share_ultras_volume_or_app_name():
    """Three Krea 2 deployments coexist; colliding names would fight."""
    assert app.APP_NAME == "redgpt2gpt-comfyui"
    assert "redgpt2gpt" in str(app.models_volume)
