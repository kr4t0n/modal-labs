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
    assert set(config) == {"darkbeast3"}
    section = config["darkbeast3"]
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


def test_the_file_id_is_what_selects_the_precision():
    """This version publishes int8, fp8, bf16, nvfp4 and int4 under one filename.

    Five files, one name, one version id. Only the file id says which is
    fetched, so it is pinned and asserted rather than left to Civitai's choice
    of primary file.
    """
    checkpoint = next(f for f in app.MODEL_FILES if isinstance(f, weights.CivitaiFile))
    assert checkpoint.model_version_id == 3173268
    assert checkpoint.file_id == 3053854
    assert f"fileId={checkpoint.file_id}" in checkpoint.url
    # The destination has to disambiguate too; upstream's name does not.
    assert "int8" in checkpoint.destination
    assert "krea2" in checkpoint.destination


def test_no_civitai_secret_is_attached():
    """Verified anonymous: a ranged GET returns 206 with real safetensors bytes.

    Notable because this listing *is* NSFW-flagged, unlike redcraft3's — so the
    flag predicts nothing and the check is the only thing that settles it.
    Asserted so the "needs no credentials" claim in the README stays honest; the
    same claim was wrong for finepornv4 and cost a release.
    """
    assert not hasattr(app, "CIVITAI_SECRET_NAME")


def test_companions_come_from_an_ungated_hugging_face_mirror():
    """Nothing here should need a token; a gated flag would break deploys."""
    hf = [f for f in app.MODEL_FILES if isinstance(f, weights.HuggingFaceFile)]
    assert len(hf) == 2
    assert all(not f.gated for f in hf)
    assert all(f.repo_id == "Comfy-Org/Krea-2" for f in hf)


def test_the_krea2_companions_match_the_ultra_service_exactly():
    """Five services, one base model, one pair of companion files.

    finepornv4, redgpt2gpt and redcraft3 carry the same assertion against
    ultra, so pinning each to ultra transitively pins all five. If these diverge
    it means one picked up a different encoder or VAE build, which changes
    output without failing.
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


def test_this_service_claims_its_own_app_and_volume():
    """Five Krea 2 deployments coexist; colliding names would fight."""
    assert app.APP_NAME == "darkbeast3-comfyui"
    assert "darkbeast3" in str(app.models_volume)
