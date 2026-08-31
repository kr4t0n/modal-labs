"""Structural tests for the Z-Image Turbo (Stable Yogi) graph."""

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
    return workflow.build_workflow(workflow.resolve_params("a test prompt", **kwargs))


def test_split_weights_are_loaded_separately():
    """Diffusion-model-only file: encoder and VAE come from their own loaders."""
    graph = graph_for()
    assert graph["load_unet"]["inputs"]["unet_name"] == workflow.DIFFUSION_MODEL
    assert graph["load_clip"]["inputs"]["clip_name"] == workflow.TEXT_ENCODER
    assert graph["load_vae"]["inputs"]["vae_name"] == workflow.VAE


def test_turbo_reference_settings_are_the_defaults():
    """Values come from ComfyUI's official Z-Image Turbo template."""
    params = workflow.resolve_params("x")
    assert (params.steps, params.cfg) == (8, 1.0)
    assert (params.sampler_name, params.scheduler) == ("res_multistep", "simple")
    assert params.shift == 3.0


def test_sampling_shift_patch_sits_between_loader_and_sampler():
    """Skipping ModelSamplingAuraFlow silently changes the noise schedule."""
    graph = graph_for()
    patch = graph["model_sampling"]
    assert patch["class_type"] == "ModelSamplingAuraFlow"
    assert patch["inputs"]["model"] == ["load_unet", 0]
    assert patch["inputs"]["shift"] == 3.0
    # The sampler must read the patched model, not the raw loader.
    assert graph["sample"]["inputs"]["model"] == ["model_sampling", 0]


def test_shift_is_overridable_and_validated():
    assert graph_for(shift=1.73)["model_sampling"]["inputs"]["shift"] == 1.73
    with pytest.raises(workflow.WorkflowError, match="shift"):
        workflow.resolve_params("x", shift=0)


def test_latent_is_the_sd3_style_sixteen_channel_one():
    """EmptyLatentImage would give a 4-channel latent and fail at sample time."""
    assert graph_for()["latent"]["class_type"] == "EmptySD3LatentImage"


def test_text_encoder_is_reached_through_the_lumina2_clip_type():
    """Z-Image subclasses Lumina2; the 8B encoders are not interchangeable."""
    inputs = graph_for()["load_clip"]["inputs"]
    assert inputs["type"] == "lumina2"
    assert inputs["clip_name"] == workflow.TEXT_ENCODER


def test_negative_defaults_to_zeroed_conditioning():
    """Matches the reference template: at cfg 1 the negative is never consulted."""
    graph = graph_for()
    node = graph[workflow.NEGATIVE_NODE_ID]
    assert node["class_type"] == "ConditioningZeroOut"
    assert node["inputs"]["conditioning"] == ["positive", 0]


def test_supplying_a_negative_swaps_in_a_real_encoder():
    graph = graph_for(negative_prompt="blurry, low quality")
    node = graph[workflow.NEGATIVE_NODE_ID]
    assert node["class_type"] == "CLIPTextEncode"
    assert node["inputs"]["text"] == "blurry, low quality"
    assert node["inputs"]["clip"] == ["load_clip", 0]
    # Either way the sampler reads the same node id.
    assert graph["sample"]["inputs"]["negative"] == [workflow.NEGATIVE_NODE_ID, 0]


def test_whitespace_only_negative_is_treated_as_absent():
    assert (
        graph_for(negative_prompt="   ")[workflow.NEGATIVE_NODE_ID]["class_type"]
        == "ConditioningZeroOut"
    )


def test_prompt_reaches_the_text_encoder():
    graph = workflow.build_workflow(workflow.resolve_params("a distinctive prompt"))
    assert graph["positive"]["inputs"]["text"] == "a distinctive prompt"
    assert graph["sample"]["inputs"]["positive"] == ["positive", 0]


def test_sampler_settings_reach_the_ksampler():
    graph = graph_for(steps=20, cfg=3.5, sampler_name="dpmpp_2m", scheduler="karras", seed=7)
    sampler = graph["sample"]["inputs"]
    assert (sampler["steps"], sampler["cfg"], sampler["seed"]) == (20, 3.5, 7)
    assert (sampler["sampler_name"], sampler["scheduler"]) == ("dpmpp_2m", "karras")


def test_sides_snap_and_reach_the_latent():
    graph = graph_for(width=1000, height=500)
    assert graph["latent"]["inputs"]["width"] == 1008
    assert graph["latent"]["inputs"]["height"] == 512


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_rejected(prompt):
    with pytest.raises(workflow.WorkflowError):
        workflow.resolve_params(prompt)


def test_every_link_targets_an_existing_node():
    for graph in (graph_for(), graph_for(negative_prompt="blurry")):
        for node_id, node in graph.items():
            for name, value in node["inputs"].items():
                if isinstance(value, list):
                    assert value[0] in graph, f"{node_id}.{name} -> missing {value[0]!r}"


def test_graph_is_json_serialisable_and_complete():
    graph = graph_for()
    json.dumps(graph)
    assert {node["class_type"] for node in graph.values()} == {
        "UNETLoader",
        "ModelSamplingAuraFlow",
        "CLIPLoader",
        "CLIPTextEncode",
        "ConditioningZeroOut",
        "EmptySD3LatentImage",
        "KSampler",
        "VAELoader",
        "VAEDecode",
        "SaveImage",
    }
    assert workflow.OUTPUT_NODE_ID in graph


def test_request_aspect_ratio_overrides_sides():
    request = server.GenerateRequest(prompt="x", width=512, height=512, aspect_ratio="16:9")
    width, height = request.dimensions()
    assert width > height and (width, height) != (512, 512)


def test_suite_imported_this_projects_modules():
    """Guards the cross-project module-name collision."""
    assert Path(workflow.__file__).resolve().parent == Path(__file__).resolve().parents[1]
    assert Path(server.__file__).resolve().parent == Path(__file__).resolve().parents[1]


GOLDEN = Path(__file__).resolve().parents[1] / "workflows" / "zimageturbostableyogi_t2i_api.json"
GOLDEN_PARAMS = {
    "prompt": "a harbour at dawn, fishing boats and pastel houses, soft light",
    "width": 1024,
    "height": 1024,
    "seed": 0,
}


def test_graph_matches_the_committed_reference():
    """Pins the emitted graph byte-for-byte against workflows/*.json."""
    built = workflow.build_workflow(workflow.resolve_params(**GOLDEN_PARAMS))
    assert built == json.loads(GOLDEN.read_text(encoding="utf-8"))
