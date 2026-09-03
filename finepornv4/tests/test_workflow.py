"""Structural tests for the FinePorn v4 graph.

Guards the things that fail silently: a sampler pairing that drifts back to the
generic turbo template, a default resolution that falls back to 1 MP, or a link
pointing at a renamed node. Whether the node schemas still match a given ComfyUI
build is answered by `client.py validate` against a live deployment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Every service ships top-level modules with these names, and pytest collects
# them all in one interpreter. Drop whatever another suite left behind so the
# imports below resolve against *this* project.
for _shared in ("workflow", "server", "app"):
    sys.modules.pop(_shared, None)

import server  # noqa: E402
import workflow  # noqa: E402


def graph_for(**kwargs):
    return workflow.build_workflow(workflow.resolve_params("a test prompt", **kwargs))


def test_sampler_defaults_come_from_the_model_card():
    """euler/beta, not the euler/simple of ComfyUI's generic turbo template.

    Drifting back to `simple` would still render, just not to the recipe its
    author published for v4 — the kind of regression nothing else would catch.
    """
    params = workflow.resolve_params("x")
    assert params.sampler_name == "euler"
    assert params.scheduler == "beta"
    assert params.cfg == 1.0
    assert params.steps == 10


def test_default_steps_sit_inside_the_published_band():
    low, high = workflow.STEPS_RANGE
    assert low <= workflow.DEFAULT_STEPS <= high


def test_defaults_render_above_one_megapixel():
    """The whole point of this service's resolution override.

    The author reports standard Krea 2 sizes underperform on this merge, so a
    silent fallback to 1024x1024 is a quality regression with no error.
    """
    params = workflow.resolve_params("x")
    assert (params.width, params.height) == (1280, 1280)
    assert params.width * params.height > 1_000_000


def test_request_model_carries_the_same_resolution_defaults():
    """The server's defaults are what actually apply; workflow's are a backstop."""
    request = server.GenerateRequest(prompt="x")
    assert request.dimensions() == (workflow.DEFAULT_SIDE, workflow.DEFAULT_SIDE)
    assert request.megapixels == workflow.DEFAULT_MEGAPIXELS


def test_aspect_ratio_keeps_the_raised_pixel_budget():
    """A ratio request must not silently drop back to the shared 1 MP budget."""
    request = server.GenerateRequest(prompt="x", aspect_ratio="3:4")
    width, height = request.dimensions()
    assert width < height
    assert width * height > 1_400_000


def test_recommended_resolutions_are_ordered_and_snapped():
    for standard, optimal, recommended in workflow.RECOMMENDED_RESOLUTIONS:
        assert standard[0] * standard[1] < optimal[0] * optimal[1]
        assert optimal[0] * optimal[1] < recommended[0] * recommended[1]
        for width, height in (standard, optimal, recommended):
            assert width % workflow.SIDE_MULTIPLE == 0
            assert height % workflow.SIDE_MULTIPLE == 0


def test_the_square_default_is_the_published_optimal():
    """1280x1280 is not invented here; it is the card's own scaling of 1024."""
    optimal_for_square = next(
        optimal
        for standard, optimal, _ in workflow.RECOMMENDED_RESOLUTIONS
        if standard == (1024, 1024)
    )
    assert optimal_for_square == (workflow.DEFAULT_SIDE, workflow.DEFAULT_SIDE)


@pytest.mark.parametrize(("requested", "expected"), [(1000, 1008), (1280, 1280), (4096, 2048)])
def test_sides_snap_to_multiples_of_16_and_clamp(requested, expected):
    assert workflow.snap_side(requested) == expected


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_rejected(prompt):
    with pytest.raises(workflow.WorkflowError):
        workflow.resolve_params(prompt)


def test_seed_is_stable_when_given_and_random_otherwise():
    assert workflow.resolve_params("x", seed=42).seed == 42
    seeds = {workflow.resolve_params("x").seed for _ in range(20)}
    assert len(seeds) > 1


def test_denoise_outside_zero_to_one_rejected():
    with pytest.raises(workflow.WorkflowError, match="denoise"):
        workflow.resolve_params("x", denoise=1.5)


def test_every_link_targets_an_existing_node():
    graph = graph_for()
    for node_id, node in graph.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                target, slot = value
                assert target in graph, f"{node_id}.{name} -> missing node {target!r}"
                assert isinstance(slot, int)


def test_graph_is_json_serialisable_and_complete():
    graph = graph_for()
    json.dumps(graph)
    assert {node["class_type"] for node in graph.values()} == {
        "UNETLoader",
        "CLIPLoader",
        "CLIPTextEncode",
        "ConditioningZeroOut",
        "EmptyLatentImage",
        "KSampler",
        "VAELoader",
        "VAEDecode",
        "SaveImage",
    }
    assert workflow.OUTPUT_NODE_ID in graph


def test_krea2_companions_are_wired_not_the_klein_ones():
    """The 8B klein encoder loads without error here and degrades output."""
    graph = graph_for()
    assert graph["load_clip"]["inputs"]["clip_name"] == workflow.TEXT_ENCODER
    assert graph["load_clip"]["inputs"]["type"] == "krea2"
    assert graph["load_vae"]["inputs"]["vae_name"] == workflow.VAE


def test_omitting_a_negative_zeroes_the_conditioning():
    node = graph_for()[workflow.NEGATIVE_NODE_ID]
    assert node["class_type"] == "ConditioningZeroOut"
    assert node["inputs"]["conditioning"] == ["positive", 0]


def test_supplying_a_negative_encodes_it_against_the_same_clip():
    graph = graph_for(negative_prompt="blurry, watermark")
    node = graph[workflow.NEGATIVE_NODE_ID]
    assert node["class_type"] == "CLIPTextEncode"
    assert node["inputs"]["text"] == "blurry, watermark"
    assert node["inputs"]["clip"] == graph["positive"]["inputs"]["clip"]


def test_suite_imported_this_projects_modules():
    """Guards the cross-project module-name collision."""
    assert Path(workflow.__file__).resolve().parent == Path(__file__).resolve().parents[1]
    assert Path(server.__file__).resolve().parent == Path(__file__).resolve().parents[1]


GOLDEN = Path(__file__).resolve().parents[1] / "workflows" / "finepornv4_krea2_t2i_api.json"
GOLDEN_PARAMS = {
    "prompt": "this is an amateur photo taken from smartphone, casual photo of a woman laughing",
    "width": 1280,
    "height": 1280,
    "seed": 0,
}


def test_graph_matches_the_committed_reference():
    """Pins the emitted graph byte-for-byte against workflows/*.json.

    That file is what users POST directly, so it must not drift; it doubles as
    an equivalence check when this module is refactored.
    """
    built = workflow.build_workflow(workflow.resolve_params(**GOLDEN_PARAMS))
    assert built == json.loads(GOLDEN.read_text(encoding="utf-8"))
