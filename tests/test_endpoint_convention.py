"""Guards the naming convention that URL derivation depends on.

Setting `MODAL_WORKSPACE` instead of a per-service `<SLUG>_MODAL_URL` works only
because every service is named consistently: directory `<slug>`, Modal app
`<slug>-comfyui`, and a class whose lowercased name is `<slug>`. Modal composes
the hostname from the last two.

Break any of those and the derivation does not fail — it returns a plausible URL
pointing at nothing, and the caller sees a connection error naming a host they
never configured. These tests are what turn that into a build failure.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from comfyui_modal import cli
from comfyui_modal.testing import install_comfyui_stubs

ROOT = Path(__file__).resolve().parents[1]


# No progress bar passed: this suite only needs the import to succeed, and
# must not clobber the recorder test_node_runtime.py installs.
install_comfyui_stubs()

from comfy_node import _runtime  # noqa: E402

# Directory name is the slug. Read from disk rather than hardcoded so a new
# service is covered the moment it exists.
SERVICES = sorted(
    path.name
    for path in ROOT.iterdir()
    if path.is_dir()
    and not path.name.startswith(".")
    and path.name != "deprecated"
    and (path / "app.py").is_file()
)


def app_source(service: str) -> str:
    return (ROOT / service / "app.py").read_text(encoding="utf-8")


def declared_app_name(service: str) -> str:
    match = re.search(r'^APP_NAME = "([^"]+)"', app_source(service), re.M)
    assert match, f"{service}/app.py declares no APP_NAME"
    return match.group(1)


def declared_web_class(service: str) -> str:
    """The class carrying the `web` asgi_app — the one Modal names the URL for."""
    tree = ast.parse(app_source(service))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(
            isinstance(item, ast.FunctionDef) and item.name == "web" for item in node.body
        ):
            return node.name
    raise AssertionError(f"{service}/app.py has no class with a `web` method")


def test_there_are_services_to_check():
    """A globbing bug that found nothing would make every test below vacuous."""
    assert len(SERVICES) >= 5


@pytest.mark.parametrize("service", SERVICES)
def test_app_name_follows_the_slug_convention(service):
    assert declared_app_name(service) == f"{service}-comfyui"


@pytest.mark.parametrize("service", SERVICES)
def test_web_class_lowercases_to_the_slug(service):
    assert declared_web_class(service).lower() == service


@pytest.mark.parametrize("service", SERVICES)
def test_client_env_var_matches_the_slug(service):
    """`derive_url` recovers the slug from the variable name, so it must match."""
    source = (ROOT / service / "client.py").read_text(encoding="utf-8")
    match = re.search(r'^ENV_URL = "([^"]+)"', source, re.M)
    assert match, f"{service}/client.py declares no ENV_URL"
    assert match.group(1) == f"{service.upper()}_MODAL_URL"


@pytest.mark.parametrize("service", SERVICES)
def test_derived_url_matches_the_documented_template(service):
    """The node package's .env.example shows each URL; derivation must agree.

    That file is what users copy, so a mismatch would hand them a working
    template alongside a derivation that points somewhere else.
    """
    example = (ROOT / "comfy_node" / ".env.example").read_text(encoding="utf-8")
    line = next(
        (line for line in example.splitlines() if line.startswith(f"{service.upper()}_MODAL_URL=")),
        None,
    )
    assert line, f"comfy_node/.env.example does not document {service}"
    documented = line.split("=", 1)[1]
    assert cli.derive_url(f"{service.upper()}_MODAL_URL", "your-workspace") == documented


def test_both_implementations_agree():
    """The node cannot import the CLI's copy, so the two are duplicated."""
    assert _runtime.MODAL_URL_SUFFIX == cli.MODAL_URL_SUFFIX
    assert _runtime.WORKSPACE_VAR == cli.WORKSPACE_VAR
    for service in SERVICES:
        var = f"{service.upper()}_MODAL_URL"
        assert _runtime.derive_url(var, "ws") == cli.derive_url(var, "ws")


def test_a_non_endpoint_variable_is_refused():
    """Deriving from an unrelated name would invent a host silently."""
    with pytest.raises(ValueError):
        cli.derive_url("MODAL_KEY", "ws")
    with pytest.raises(RuntimeError):
        _runtime.derive_url("MODAL_KEY", "ws")


def test_a_non_conforming_variable_still_gets_the_helpful_message(monkeypatch):
    """A workspace set in the shell must not hijack an unrelated variable.

    Deriving from `DEMO_URL` would report "cannot derive a URL" — an error
    about a mechanism the caller was not using — instead of naming the variable
    they actually need to set.
    """
    monkeypatch.setenv("MODAL_WORKSPACE", "acme")
    monkeypatch.delenv("DEMO_URL", raising=False)
    monkeypatch.setattr(_runtime, "_dotenv", dict)
    with pytest.raises(RuntimeError, match="DEMO_URL"):
        _runtime.endpoint("", "DEMO_URL")


def test_explicit_url_beats_the_workspace(monkeypatch):
    monkeypatch.setenv("MODAL_WORKSPACE", "ws")
    monkeypatch.setenv("DEMO_MODAL_URL", "https://explicit.modal.run/")
    assert cli.endpoint(None, "DEMO_MODAL_URL") == "https://explicit.modal.run"


def test_workspace_used_when_the_service_variable_is_unset(monkeypatch):
    monkeypatch.delenv("DEMO_MODAL_URL", raising=False)
    monkeypatch.setenv("MODAL_WORKSPACE", "acme")
    assert cli.endpoint(None, "DEMO_MODAL_URL") == "https://acme--demo-comfyui-demo-web.modal.run"


def test_node_falls_back_to_the_workspace(monkeypatch):
    monkeypatch.setattr(_runtime, "_dotenv", dict)
    monkeypatch.delenv("DEMO_MODAL_URL", raising=False)
    monkeypatch.setenv("MODAL_WORKSPACE", "acme")
    assert (
        _runtime.endpoint("", "DEMO_MODAL_URL") == "https://acme--demo-comfyui-demo-web.modal.run"
    )


def test_node_error_names_both_ways_out(monkeypatch):
    monkeypatch.setattr(_runtime, "_dotenv", dict)
    monkeypatch.delenv("DEMO_MODAL_URL", raising=False)
    monkeypatch.delenv("MODAL_WORKSPACE", raising=False)
    with pytest.raises(RuntimeError, match=r"DEMO_MODAL_URL.*MODAL_WORKSPACE"):
        _runtime.endpoint("", "DEMO_MODAL_URL")
