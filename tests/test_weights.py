"""Tests for the shared weight-fetching layer.

The network is never touched: each test substitutes a fake `fetch`, so what is
under test is the placement, idempotence and digest logic rather than any
particular host.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from comfyui_modal import weights


class FakeFile:
    """A weight file whose fetch just writes known bytes into staging."""

    def __init__(self, destination: str, payload: bytes = b"weights"):
        self.destination = destination
        self.payload = payload
        self.fetches = 0

    def fetch(self, staging_dir: str) -> str:
        self.fetches += 1
        staged = Path(staging_dir) / Path(self.destination).name
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(self.payload)
        return str(staged)


def test_files_land_at_their_destinations(tmp_path):
    files = (FakeFile("diffusion_models/a.safetensors"), FakeFile("vae/b.safetensors"))
    written = weights.download_weights(files, str(tmp_path))

    assert (tmp_path / "diffusion_models/a.safetensors").read_bytes() == b"weights"
    assert (tmp_path / "vae/b.safetensors").is_file()
    assert written == [str(tmp_path / f.destination) for f in files]


def test_existing_files_are_left_alone(tmp_path):
    """Re-running must not refetch tens of gigabytes."""
    target = tmp_path / "vae/b.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already here")

    fake = FakeFile("vae/b.safetensors")
    weights.download_weights((fake,), str(tmp_path))

    assert fake.fetches == 0
    assert target.read_bytes() == b"already here"


def test_force_refetches(tmp_path):
    target = tmp_path / "vae/b.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"stale")

    fake = FakeFile("vae/b.safetensors", payload=b"fresh")
    weights.download_weights((fake,), str(tmp_path), force=True)

    assert fake.fetches == 1
    assert target.read_bytes() == b"fresh"


def test_staging_is_cleaned_up(tmp_path):
    weights.download_weights((FakeFile("vae/b.safetensors"),), str(tmp_path))
    assert not (tmp_path / ".staging").exists()


def test_destinations_helper():
    files = (FakeFile("a/one.safetensors"), FakeFile("b/two.safetensors"))
    assert weights.destinations(files) == ("a/one.safetensors", "b/two.safetensors")


def test_civitai_url_pins_the_version():
    """A floating 'latest' URL would silently change the model under us."""
    civitai = weights.CivitaiFile(
        model_version_id=123, file_id=456, destination="x/y.safetensors", sha256="ab" * 32
    )
    assert civitai.url == "https://civitai.com/api/download/models/123?fileId=456"
    # Always the official host, whatever a listing page was reached through.
    assert civitai.url.startswith("https://civitai.com/")


def test_civitai_digest_mismatch_refuses_to_install(tmp_path, monkeypatch):
    """The digest is the only integrity check on a third-party CDN download."""
    payload = b"substituted weights"
    civitai = weights.CivitaiFile(
        model_version_id=1,
        file_id=2,
        destination="diffusion_models/m.safetensors",
        sha256=hashlib.sha256(b"the real weights").hexdigest(),
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def iter_bytes(self, size):
            yield payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("httpx.stream", lambda *a, **k: FakeResponse())

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        weights.download_weights((civitai,), str(tmp_path))

    # Nothing half-written is left behind for ComfyUI to load.
    assert not (tmp_path / civitai.destination).exists()


def test_civitai_accepts_a_matching_digest(tmp_path, monkeypatch):
    payload = b"the real weights"
    civitai = weights.CivitaiFile(
        model_version_id=1,
        file_id=2,
        destination="diffusion_models/m.safetensors",
        sha256=hashlib.sha256(payload).hexdigest().upper(),  # case must not matter
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def iter_bytes(self, size):
            yield payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("httpx.stream", lambda *a, **k: FakeResponse())
    weights.download_weights((civitai,), str(tmp_path))
    assert (tmp_path / civitai.destination).read_bytes() == payload
