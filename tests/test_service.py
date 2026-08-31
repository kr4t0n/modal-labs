"""Tests for the Modal-side plumbing every service shares.

These target failures that otherwise surface minutes into a remote image build,
or at model-load time inside a running container.
"""

from __future__ import annotations

import re

import pytest
import yaml

from comfyui_modal import service


def test_build_commands_are_single_line():
    """A multi-line RUN entry produces a Dockerfile that will not parse.

    This is the guard for a real failure: interpolating a YAML document into a
    `run_commands` string once broke a deploy several minutes into the build.
    """
    with pytest.raises(ValueError, match="spans multiple lines"):
        service.single_line("echo one", "printf 'a\nb' > /tmp/x")

    assert service.single_line("echo one", "echo two") == ("echo one", "echo two")


def test_extra_model_paths_yaml_is_valid_and_rooted_at_the_volume():
    text = service.extra_model_paths_yaml("demo", ("diffusion_models", "text_encoders", "vae"))
    config = yaml.safe_load(text)
    assert set(config) == {"demo"}
    section = config["demo"]
    assert section["base_path"] == service.MODELS_DIR
    # These keys are ComfyUI's own folder names. A typo yields an empty model
    # list and an error only once a prompt is queued.
    assert section["diffusion_models"] == "diffusion_models"
    assert section["text_encoders"] == "text_encoders"
    assert section["vae"] == "vae"


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("DEMO_GPU", raising=False)
    settings = service.Settings.from_env("DEMO")
    assert settings.gpu == "H100"
    # One container by default: ComfyUI's queue, /history and /view are
    # per-container state.
    assert settings.max_containers == 1
    assert settings.require_auth is True
    assert settings.ui_require_auth is False


def test_settings_read_the_prefixed_environment(monkeypatch):
    monkeypatch.setenv("DEMO_GPU", "L40S")
    monkeypatch.setenv("DEMO_MAX_CONTAINERS", "3")
    monkeypatch.setenv("DEMO_REQUIRE_AUTH", "0")
    settings = service.Settings.from_env("DEMO")
    assert (settings.gpu, settings.max_containers, settings.require_auth) == ("L40S", 3, False)


def test_settings_prefixes_do_not_leak(monkeypatch):
    """Two services read from one environment; one must not see the other's."""
    monkeypatch.setenv("OTHER_GPU", "L4")
    assert service.Settings.from_env("DEMO").gpu == "H100"


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_env_flag_falsey_spellings(monkeypatch, value):
    monkeypatch.setenv("DEMO_FLAG", value)
    assert service.env_flag("DEMO_FLAG", "1") is False


def test_assert_models_present_names_what_is_missing():
    with pytest.raises(RuntimeError, match=re.escape("definitely-not-here.safetensors")):
        service.assert_models_present(["definitely-not-here.safetensors"])
