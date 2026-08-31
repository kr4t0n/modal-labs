"""Tests for the resolution and seed arithmetic every service shares."""

from __future__ import annotations

import pytest

from comfyui_modal import geometry


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(1000, 1008), (1024, 1024), (1, 256), (4096, 2048), (2041, 2048)],
)
def test_sides_snap_to_multiples_of_16_and_clamp(requested, expected):
    assert geometry.snap_side(requested) == expected


def test_resolution_for_respects_ratio_and_budget():
    width, height = geometry.resolution_for("16:9", 1.0)
    assert width % 16 == 0 and height % 16 == 0
    assert 1.7 < width / height < 1.8
    assert 0.8e6 < width * height < 1.3e6


def test_megapixels_scale_the_result():
    small = geometry.resolution_for("1:1", 1.0)
    large = geometry.resolution_for("1:1", 4.0)
    assert large[0] > small[0] and large[1] > small[1]


@pytest.mark.parametrize("bad", ["5:1", "", "16;9"])
def test_unknown_aspect_ratio_rejected(bad):
    with pytest.raises(geometry.WorkflowError):
        geometry.resolution_for(bad)


def test_non_positive_megapixels_rejected():
    with pytest.raises(geometry.WorkflowError):
        geometry.resolution_for("1:1", 0)


def test_seed_normalisation():
    assert geometry.normalise_seed(42) == 42
    assert geometry.normalise_seed(geometry.MAX_SEED + 1) == 0
    seeds = {geometry.normalise_seed(None) for _ in range(20)}
    assert len(seeds) > 1
    assert all(0 <= seed <= geometry.MAX_SEED for seed in seeds)
