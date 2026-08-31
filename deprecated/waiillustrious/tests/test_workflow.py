"""Structural tests for the WAI-illustrious-SDXL graph."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Every service ships top-level modules with these names, and pytest collects
# them all in one interpreter. Drop whatever another project's suite left in
# sys.modules so the imports below resolve against *this* project.
for _shared in ("workflow", "server", "app"):
    sys.modules.pop(_shared, None)

import server  # noqa: E402
import workflow  # noqa: E402


def graph_for(**kwargs):
    return workflow.build_workflow(workflow.resolve_params("1girl, solo", **kwargs))


def test_single_checkpoint_supplies_model_clip_and_vae():
    """The SDXL shape: one loader feeds all three, unlike the other services."""
    graph = graph_for()
    assert sum(n["class_type"] == "CheckpointLoaderSimple" for n in graph.values()) == 1
    assert graph["sample"]["inputs"]["model"] == ["load_checkpoint", 0]
    assert graph["clip_skip"]["inputs"]["clip"] == ["load_checkpoint", 1]
    assert graph["decode"]["inputs"]["vae"] == ["load_checkpoint", 2]


def test_both_encoders_read_the_clip_skipped_clip():
    """Skipping for one prompt but not the other would silently skew guidance."""
    graph = graph_for()
    assert graph["positive"]["inputs"]["clip"] == ["clip_skip", 0]
    assert graph["negative"]["inputs"]["clip"] == ["clip_skip", 0]
    assert graph["clip_skip"]["inputs"]["stop_at_clip_layer"] == -2


def test_prompts_reach_their_own_encoders():
    params = workflow.resolve_params("1girl, masterpiece", negative_prompt="blurry")
    graph = workflow.build_workflow(params)
    assert graph["positive"]["inputs"]["text"] == "1girl, masterpiece"
    assert graph["negative"]["inputs"]["text"] == "blurry"
    assert graph["sample"]["inputs"]["positive"] == ["positive", 0]
    assert graph["sample"]["inputs"]["negative"] == ["negative", 0]


def test_default_negative_is_applied_but_can_be_cleared():
    assert workflow.resolve_params("x").negative_prompt == workflow.DEFAULT_NEGATIVE_PROMPT
    assert workflow.resolve_params("x", negative_prompt="").negative_prompt == ""


@pytest.mark.parametrize("clip_skip", [0, 1, -25, -30])
def test_out_of_range_clip_skip_rejected(clip_skip):
    """ComfyUI accepts -24..-1; anything else fails obscurely at queue time."""
    with pytest.raises(workflow.WorkflowError, match="clip_skip"):
        workflow.resolve_params("x", clip_skip=clip_skip)


@pytest.mark.parametrize("denoise", [-0.1, 1.5])
def test_out_of_range_denoise_rejected(denoise):
    with pytest.raises(workflow.WorkflowError):
        workflow.resolve_params("x", denoise=denoise)


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_rejected(prompt):
    with pytest.raises(workflow.WorkflowError):
        workflow.resolve_params(prompt)


def test_sampler_settings_reach_the_ksampler():
    graph = graph_for(steps=32, cfg=7.5, sampler_name="dpmpp_2m", scheduler="karras", seed=99)
    sampler = graph["sample"]["inputs"]
    assert (sampler["steps"], sampler["cfg"], sampler["seed"]) == (32, 7.5, 99)
    assert (sampler["sampler_name"], sampler["scheduler"]) == ("dpmpp_2m", "karras")
    assert sampler["denoise"] == 1.0


def test_sides_snap_and_reach_the_latent():
    graph = graph_for(width=830, height=1210)
    # Snapped up to multiples of 16, which are also valid multiples of 8.
    assert graph["latent"]["inputs"]["width"] == 832
    assert graph["latent"]["inputs"]["height"] == 1216
    assert graph["latent"]["inputs"]["width"] % 8 == 0


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
        "CheckpointLoaderSimple",
        "CLIPSetLastLayer",
        "CLIPTextEncode",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }
    assert workflow.OUTPUT_NODE_ID in graph


def test_seed_is_stable_when_given_and_random_otherwise():
    assert workflow.resolve_params("x", seed=42).seed == 42
    assert len({workflow.resolve_params("x").seed for _ in range(20)}) > 1


def test_request_aspect_ratio_overrides_sides():
    request = server.GenerateRequest(prompt="x", width=512, height=512, aspect_ratio="2:3")
    width, height = request.dimensions()
    assert height > width and (width, height) != (512, 512)


def test_suite_imported_this_projects_modules():
    """Guards the cross-project module-name collision."""
    assert Path(workflow.__file__).resolve().parent == Path(__file__).resolve().parents[1]
    assert Path(server.__file__).resolve().parent == Path(__file__).resolve().parents[1]


GOLDEN = Path(__file__).resolve().parents[1] / "workflows" / "wai_illustrious_sdxl_t2i_api.json"
GOLDEN_PARAMS = {
    "prompt": "1girl, solo, silver hair, red eyes, city at night, masterpiece, best quality",
    "width": 832,
    "height": 1216,
    "seed": 0,
}


def test_graph_matches_the_committed_reference():
    """Pins the emitted graph byte-for-byte against workflows/*.json."""
    built = workflow.build_workflow(workflow.resolve_params(**GOLDEN_PARAMS))
    assert built == json.loads(GOLDEN.read_text(encoding="utf-8"))
