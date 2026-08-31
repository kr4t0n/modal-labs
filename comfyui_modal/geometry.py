"""Resolution and seed arithmetic shared by every model graph.

Both services target ComfyUI's FLUX.2-style latents, which pack 16x16 pixel
patches, so the snapping rule and the aspect-ratio table are identical. Keeping
one copy means a change to the trained resolution range cannot apply to one
service and silently not the other.
"""

from __future__ import annotations

import math
import random

# Sides must be multiples of 16, within the range the models are trained for.
MIN_SIDE = 256
MAX_SIDE = 2048
SIDE_MULTIPLE = 16

MAX_SEED = 0xFFFFFFFFFFFFFFFF

# Mirrors ComfyUI's ResolutionSelector node, so the ratios a remote API offers
# are the ones the local UI offers.
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "2:3": (2, 3),
    "3:2": (3, 2),
    "3:4": (3, 4),
    "4:3": (4, 3),
    "9:16": (9, 16),
    "16:9": (16, 9),
    "21:9": (21, 9),
}


class WorkflowError(ValueError):
    """Raised when generation parameters cannot produce a valid graph."""


def snap_side(value: int) -> int:
    """Round a side up to a multiple of 16, clamped to the trained range.

    Matches the `max(((a + 15) // 16) * 16, 256)` expression in ComfyUI's own
    templates, then clamps so a typo cannot request an 8k latent.
    """
    snapped = max(((int(value) + SIDE_MULTIPLE - 1) // SIDE_MULTIPLE) * SIDE_MULTIPLE, MIN_SIDE)
    return min(snapped, MAX_SIDE)


def resolution_for(aspect_ratio: str, megapixels: float = 1.0) -> tuple[int, int]:
    """Width/height for an aspect ratio at a pixel budget, snapped for the model."""
    if aspect_ratio not in ASPECT_RATIOS:
        raise WorkflowError(
            f"unknown aspect_ratio {aspect_ratio!r}; expected one of {sorted(ASPECT_RATIOS)}"
        )
    if megapixels <= 0:
        raise WorkflowError("megapixels must be positive")
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    scale = math.sqrt(megapixels * 1024 * 1024 / (w_ratio * h_ratio))
    return snap_side(round(w_ratio * scale)), snap_side(round(h_ratio * scale))


def random_seed() -> int:
    return random.randrange(MAX_SEED + 1)


def normalise_seed(seed: int | None) -> int:
    """A caller's seed, wrapped into range — or a fresh one when omitted."""
    return random_seed() if seed is None else int(seed) % (MAX_SEED + 1)
