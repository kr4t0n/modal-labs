"""Fetching model weights into a Modal Volume.

Two sources recur across services and neither is quite like the other:

* **Hugging Face** — addressed by repo and path, sometimes gated behind a token
  and an accepted licence. Integrity is handled by the hub client.
* **Civitai** — addressed by a model-version id, answered with a short-lived
  presigned CDN redirect, and with no integrity guarantee of its own. The
  published SHA256 is therefore verified here, and a mismatch refuses to install.

Both land in a staging directory on the same Volume and are renamed into place,
so installing a 14 GB file is a rename rather than a second copy. Downloads are
idempotent by destination: an existing file is left alone unless `force`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import NamedTuple, Protocol


class WeightFile(Protocol):
    """A file to place at `destination`, relative to the models directory."""

    destination: str

    def fetch(self, staging_dir: str) -> str:
        """Download to staging and return the path of the downloaded file."""


class HuggingFaceFile(NamedTuple):
    """One file from a Hugging Face repo."""

    repo_id: str
    filename: str
    destination: str
    # Gated repos need HF_TOKEN in the environment *and* an accepted licence;
    # flagging them lets the error message say which.
    gated: bool = False

    def fetch(self, staging_dir: str) -> str:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError

        try:
            return hf_hub_download(
                repo_id=self.repo_id, filename=self.filename, local_dir=staging_dir
            )
        except GatedRepoError as exc:
            raise RuntimeError(
                f"{self.repo_id} is gated. Accept the licence at "
                f"https://huggingface.co/{self.repo_id} using the account that owns "
                "HF_TOKEN, then re-run."
            ) from exc


class CivitaiFile(NamedTuple):
    """One file from Civitai, pinned by version and verified by digest."""

    model_version_id: int
    file_id: int
    destination: str
    sha256: str

    @property
    def url(self) -> str:
        # Civitai answers this with a short-lived presigned CDN link, so the
        # redirect has to be followed at download time rather than pinned.
        return (
            f"https://civitai.com/api/download/models/{self.model_version_id}?fileId={self.file_id}"
        )

    def fetch(self, staging_dir: str) -> str:
        import httpx

        staged = Path(staging_dir) / Path(self.destination).name
        staged.parent.mkdir(parents=True, exist_ok=True)

        headers = {}
        # Nothing sets this today — the downloads are anonymous. It exists so
        # that if Civitai starts gating, attaching a Modal Secret is the whole
        # fix, with no code change.
        if token := os.environ.get("CIVITAI_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"

        digest = hashlib.sha256()
        with httpx.stream(
            "GET",
            self.url,
            follow_redirects=True,
            headers=headers,
            timeout=httpx.Timeout(connect=30.0, read=None, write=None, pool=None),
        ) as response:
            response.raise_for_status()
            with staged.open("wb") as handle:
                for chunk in response.iter_bytes(4 << 20):
                    handle.write(chunk)
                    digest.update(chunk)

        actual = digest.hexdigest()
        if actual != self.sha256.lower():
            staged.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {self.destination}: expected "
                f"{self.sha256.lower()}, got {actual}. Refusing to install."
            )
        return str(staged)


def download_weights(
    files: tuple[WeightFile, ...], models_dir: str, *, force: bool = False
) -> list[str]:
    """Fetch every file that is not already in place, and return their paths."""
    staging_dir = f"{models_dir}/.staging"
    written = []

    for weight in files:
        target = Path(models_dir, weight.destination)
        if target.is_file() and not force:
            print(f"have {weight.destination}, skipping")
            written.append(str(target))
            continue

        print(f"fetching {weight.destination} ...")
        staged = weight.fetch(staging_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Same Volume, so this is a rename rather than a second copy.
        os.replace(staged, target)
        written.append(str(target))

    shutil.rmtree(staging_dir, ignore_errors=True)
    return written


def destinations(files: tuple[WeightFile, ...]) -> tuple[str, ...]:
    """The relative paths a service needs present before ComfyUI can start."""
    return tuple(weight.destination for weight in files)
