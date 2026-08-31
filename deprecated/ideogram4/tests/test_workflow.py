"""Structural tests for the Ideogram 4 graph.

These guard the parts that fail silently: a preset that stops being applied, a
dimension that stops snapping to 16, or a link that points at a node that was
renamed. Whether the *node schemas* still match a given ComfyUI build is a
different question, answered by `client.py validate` against a live deployment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Both services ship top-level modules with these names, and pytest collects
# every suite in one interpreter. Drop whatever the other project's suite left
# in sys.modules so the imports below resolve against *this* project.
for _shared in ("workflow", "server", "app"):
    sys.modules.pop(_shared, None)

import server  # noqa: E402
import workflow  # noqa: E402


def graph_for(**kwargs):
    return workflow.build_workflow(workflow.resolve_params("a test prompt", **kwargs))


def test_preset_supplies_schedule():
    params = workflow.resolve_params("x", preset="Turbo")
    assert (params.steps, params.mu, params.std) == (12, 0.5, 1.75)


def test_explicit_values_beat_the_preset():
    params = workflow.resolve_params("x", preset="Turbo", steps=30, std=1.5)
    assert params.steps == 30
    assert params.std == 1.5
    assert params.mu == 0.5  # untouched, still from the preset


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(1000, 1008), (1024, 1024), (1, 256), (4096, 2048), (2041, 2048)],
)
def test_sides_snap_to_multiples_of_16_and_clamp(requested, expected):
    assert workflow.snap_side(requested) == expected


def test_resolution_for_respects_ratio_and_budget():
    width, height = workflow.resolution_for("16:9", 1.0)
    assert width % 16 == 0 and height % 16 == 0
    assert 1.7 < width / height < 1.8
    assert 0.8e6 < width * height < 1.3e6


def test_unknown_aspect_ratio_rejected():
    with pytest.raises(workflow.WorkflowError):
        workflow.resolution_for("5:1")


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_rejected(prompt):
    with pytest.raises(workflow.WorkflowError):
        workflow.resolve_params(prompt)


def test_unknown_preset_rejected():
    with pytest.raises(workflow.WorkflowError):
        workflow.resolve_params("x", preset="Ludicrous")


def test_seed_is_stable_when_given_and_random_otherwise():
    assert workflow.resolve_params("x", seed=42).seed == 42
    seeds = {workflow.resolve_params("x").seed for _ in range(20)}
    assert len(seeds) > 1
    assert all(0 <= seed <= workflow.MAX_SEED for seed in seeds)


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
    classes = {node["class_type"] for node in graph.values()}
    assert classes == {
        "UNETLoader",
        "CFGOverride",
        "CLIPLoader",
        "CLIPTextEncode",
        "ConditioningZeroOut",
        "DualModelGuider",
        "KSamplerSelect",
        "Ideogram4Scheduler",
        "RandomNoise",
        "EmptyFlux2LatentImage",
        "SamplerCustomAdvanced",
        "VAELoader",
        "VAEDecode",
        "SaveImage",
    }
    assert workflow.OUTPUT_NODE_ID in graph


def test_prompt_text_reaches_the_text_encoder():
    """The one thing whose absence yields a good image that ignores the prompt."""
    graph = workflow.build_workflow(workflow.resolve_params("a distinctive prompt"))
    assert graph["positive"]["inputs"]["text"] == "a distinctive prompt"
    assert graph["positive"]["inputs"]["clip"] == ["load_clip", 0]
    assert graph["guider"]["inputs"]["positive"] == ["positive", 0]


def test_structured_caption_is_serialised_into_the_encoder():
    request = server.GenerateRequest(json_prompt={"high_level_description": "a bee"})
    graph = workflow.build_workflow(workflow.resolve_params(request.caption()))
    assert "high_level_description" in graph["positive"]["inputs"]["text"]
    assert "a bee" in graph["positive"]["inputs"]["text"]


def test_dual_branch_wiring():
    """The late-CFG override must sit between the conditional UNet and guider."""
    graph = graph_for()
    assert graph["late_cfg"]["inputs"]["model"] == ["load_unet", 0]
    guider = graph["guider"]["inputs"]
    assert guider["model"] == ["late_cfg", 0]
    assert guider["model_negative"] == ["load_unet_uncond", 0]
    assert guider["negative"] == ["negative", 0]
    assert graph["negative"]["inputs"]["conditioning"] == ["positive", 0]


def test_scheduler_sees_the_snapped_dimensions():
    graph = graph_for(width=1000, height=500)
    assert graph["sigmas"]["inputs"]["width"] == graph["latent"]["inputs"]["width"] == 1008
    assert graph["sigmas"]["inputs"]["height"] == graph["latent"]["inputs"]["height"] == 512


def test_request_prefers_structured_caption():
    request = server.GenerateRequest(prompt="ignored", json_prompt={"a": 1})
    assert json.loads(request.caption()) == {"a": 1}


def test_request_aspect_ratio_overrides_sides():
    request = server.GenerateRequest(prompt="x", width=512, height=512, aspect_ratio="21:9")
    width, height = request.dimensions()
    assert width > height
    assert (width, height) != (512, 512)


def test_suite_imported_this_projects_modules():
    """Guards the cross-project module-name collision.

    Both services ship a top-level `workflow`; if the sys.modules reset at the
    top of this file were dropped, this suite would silently exercise the other
    project's graph and most assertions would still pass.
    """
    assert Path(workflow.__file__).resolve().parent == Path(__file__).resolve().parents[1]
    assert Path(server.__file__).resolve().parent == Path(__file__).resolve().parents[1]


GOLDEN = Path(__file__).resolve().parents[1] / "workflows" / "ideogram4_t2i_api.json"
GOLDEN_PARAMS = {
    "prompt": "a vintage travel poster for the rings of Saturn, bold type reading 'SATURN'",
    "width": 1024,
    "height": 1024,
    "seed": 0,
    "preset": "Default",
}


def test_graph_matches_the_committed_reference():
    """Pins the emitted graph byte-for-byte against workflows/*.json.

    The reference file is what users POST directly, so it must not drift. It
    doubles as an equivalence check when this module is refactored.
    """
    built = workflow.build_workflow(workflow.resolve_params(**GOLDEN_PARAMS))
    assert built == json.loads(GOLDEN.read_text(encoding="utf-8"))
