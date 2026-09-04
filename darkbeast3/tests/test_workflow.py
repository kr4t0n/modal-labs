"""Structural tests for the Dark Beast v3 graph.

Guards the things that fail silently: a companion file that drifts away from the
other Krea 2 services, sampler defaults imported from the H3 video edition on
the same listing, or a link pointing at a renamed node. Whether the node schemas
still match a given ComfyUI build is answered by `client.py validate` against a
live deployment.
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


def test_sampler_defaults_are_the_turbo_template_values():
    """This version publishes no settings, so these are ComfyUI's.

    Specifically *not* the "6-10 steps" the model description gives: that is
    the H3 video edition's figure, from a different base model on the same
    listing page, and importing it would be borrowing a number from elsewhere.
    """
    params = workflow.resolve_params("x")
    assert params.sampler_name == "euler"
    assert params.scheduler == "simple"
    assert params.steps == 8
    assert params.cfg == 1.0


def test_defaults_source_admits_it_is_a_template():
    """A caller should be able to tell a template default from a published one."""
    assert "template" in workflow.DEFAULTS_SOURCE
    assert "no settings" in workflow.DEFAULTS_SOURCE


def test_the_loaded_file_names_its_precision_and_base():
    """Upstream gives five precisions one name that says neither."""
    assert "int8" in workflow.DIFFUSION_MODEL
    assert "krea2" in workflow.DIFFUSION_MODEL
    assert graph_for()["load_unet"]["inputs"]["unet_name"] == workflow.DIFFUSION_MODEL


def test_default_resolution_is_the_shared_one_megapixel():
    """Nothing upstream asks for a raised budget here, unlike finepornv4."""
    params = workflow.resolve_params("x")
    assert (params.width, params.height) == (1024, 1024)
    assert server.GenerateRequest(prompt="x").dimensions() == (1024, 1024)


@pytest.mark.parametrize(("requested", "expected"), [(1000, 1008), (1024, 1024), (4096, 2048)])
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


def test_this_is_a_still_image_graph():
    """The listing's headline product is a video pipeline; this is not it.

    A single latent, a single sampler, one SaveImage — no frame count, no
    interpolation. Anything resembling the H3 short-film pipeline appearing here
    would mean the wrong base model was wired up.
    """
    graph = graph_for()
    assert graph["latent"]["class_type"] == "EmptyLatentImage"
    assert "batch_size" in graph["latent"]["inputs"]
    assert sum(n["class_type"] == "KSampler" for n in graph.values()) == 1
    assert graph[workflow.OUTPUT_NODE_ID]["class_type"] == "SaveImage"


def test_krea2_companions_are_wired_not_the_klein_ones():
    """The listing also has FLUX.2 klein versions, whose encoder differs."""
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


GOLDEN = Path(__file__).resolve().parents[1] / "workflows" / "darkbeast3_krea2_t2i_api.json"
GOLDEN_PARAMS = {
    "prompt": "a close-up portrait in hard directional light, heavy film grain",
    "width": 1024,
    "height": 1024,
    "seed": 0,
}


def test_graph_matches_the_committed_reference():
    """Pins the emitted graph byte-for-byte against workflows/*.json.

    That file is what users POST directly, so it must not drift; it doubles as
    an equivalence check when this module is refactored.
    """
    built = workflow.build_workflow(workflow.resolve_params(**GOLDEN_PARAMS))
    assert built == json.loads(GOLDEN.read_text(encoding="utf-8"))
