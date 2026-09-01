"""Structural tests for the FLUX.2 klein graph.

Same intent as the ideogram4 suite: guard the things that fail silently — a
variant whose sampler defaults stop being applied, a dimension that stops
snapping to 16, or a link pointing at a renamed node. Whether the node schemas
still match a given ComfyUI build is answered by `client.py validate` against a
live deployment.
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


@pytest.mark.parametrize(
    ("variant", "checkpoint", "steps", "cfg"),
    [
        ("base", "flux-2-klein-base-9b-fp8.safetensors", 20, 5.0),
        ("distilled", "flux-2-klein-9b-fp8.safetensors", 4, 1.0),
        ("ponpoke-uncensored", "flux-2-klein-base-9b-fp8.safetensors", 20, 5.0),
    ],
)
def test_variant_selects_checkpoint_and_sampler_defaults(variant, checkpoint, steps, cfg):
    """Values come from the official templates; drifting from them is a bug."""
    params = workflow.resolve_params("x", variant=variant)
    assert params.checkpoint == checkpoint
    assert params.steps == steps
    assert params.cfg == cfg


def test_only_the_uncensored_variant_swaps_the_text_encoder():
    """The encoder is per-variant; a leak either way changes what renders."""
    graphs = {
        name: workflow.build_workflow(workflow.resolve_params("x", variant=name))
        for name in workflow.VARIANTS
    }
    encoders = {n: g["load_clip"]["inputs"]["clip_name"] for n, g in graphs.items()}
    assert encoders["base"] == encoders["distilled"] == workflow.TEXT_ENCODER
    assert encoders["ponpoke-uncensored"] == workflow.UNCENSORED_TEXT_ENCODER

    # Same transformer and schedule as base: the encoder is the only difference.
    assert (
        graphs["ponpoke-uncensored"]["load_unet"]["inputs"] == graphs["base"]["load_unet"]["inputs"]
    )
    assert graphs["ponpoke-uncensored"]["sigmas"]["inputs"] == graphs["base"]["sigmas"]["inputs"]


def test_explicit_values_beat_the_variant():
    params = workflow.resolve_params("x", variant="distilled", steps=8, cfg=2.5)
    assert params.steps == 8
    assert params.cfg == 2.5
    assert params.checkpoint == "flux-2-klein-9b-fp8.safetensors"


def test_unknown_variant_rejected():
    with pytest.raises(workflow.WorkflowError):
        workflow.resolve_params("x", variant="turbo")


@pytest.mark.parametrize(
    ("requested", "expected"), [(1000, 1008), (1024, 1024), (1, 256), (4096, 2048)]
)
def test_sides_snap_to_multiples_of_16_and_clamp(requested, expected):
    assert workflow.snap_side(requested) == expected


def test_resolution_for_respects_ratio_and_budget():
    width, height = workflow.resolution_for("16:9", 1.0)
    assert width % 16 == 0 and height % 16 == 0
    assert 1.7 < width / height < 1.8


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_rejected(prompt):
    with pytest.raises(workflow.WorkflowError):
        workflow.resolve_params(prompt)


def test_seed_is_stable_when_given_and_random_otherwise():
    assert workflow.resolve_params("x", seed=42).seed == 42
    seeds = {workflow.resolve_params("x").seed for _ in range(20)}
    assert len(seeds) > 1


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
        "CFGGuider",
        "KSamplerSelect",
        "Flux2Scheduler",
        "RandomNoise",
        "EmptyFlux2LatentImage",
        "SamplerCustomAdvanced",
        "VAELoader",
        "VAEDecode",
        "SaveImage",
    }
    assert workflow.OUTPUT_NODE_ID in graph


def test_prompt_and_negative_reach_separate_encoders():
    """The failure mode this guards: a good image that ignores the prompt."""
    params = workflow.resolve_params("a distinctive prompt", negative_prompt="blurry, jpeg")
    graph = workflow.build_workflow(params)
    assert graph["positive"]["inputs"]["text"] == "a distinctive prompt"
    assert graph["negative"]["inputs"]["text"] == "blurry, jpeg"
    # Both encoders must share the one loaded text encoder.
    assert graph["positive"]["inputs"]["clip"] == graph["negative"]["inputs"]["clip"]
    assert graph["guider"]["inputs"]["positive"] == ["positive", 0]
    assert graph["guider"]["inputs"]["negative"] == ["negative", 0]


def test_single_transformer_wiring():
    """Unlike Ideogram 4 there is no unconditional model; CFGGuider takes one."""
    graph = graph_for()
    assert graph["guider"]["inputs"]["model"] == ["load_unet", 0]
    assert "model_negative" not in graph["guider"]["inputs"]
    assert sum(n["class_type"] == "UNETLoader" for n in graph.values()) == 1


def test_scheduler_and_latent_agree_on_dimensions():
    graph = graph_for(width=1000, height=500)
    assert graph["sigmas"]["inputs"]["width"] == graph["latent"]["inputs"]["width"] == 1008
    assert graph["sigmas"]["inputs"]["height"] == graph["latent"]["inputs"]["height"] == 512


def test_request_aspect_ratio_overrides_sides():
    request = server.GenerateRequest(prompt="x", width=512, height=512, aspect_ratio="21:9")
    width, height = request.dimensions()
    assert width > height and (width, height) != (512, 512)


def test_request_defaults_leave_sampler_to_the_variant():
    request = server.GenerateRequest(prompt="x", variant="distilled")
    assert request.steps is None and request.cfg is None


def test_suite_imported_this_projects_modules():
    """Guards the cross-project module-name collision.

    Both services ship a top-level `workflow`; if the sys.modules reset at the
    top of this file were dropped, this suite would silently exercise the other
    project's graph and most assertions would still pass.
    """
    assert Path(workflow.__file__).resolve().parent == Path(__file__).resolve().parents[1]
    assert Path(server.__file__).resolve().parent == Path(__file__).resolve().parents[1]


GOLDEN = Path(__file__).resolve().parents[1] / "workflows" / "flux2_klein_9b_t2i_api.json"
GOLDEN_PARAMS = {
    "prompt": "a vintage motorcycle parked in front of a retro diner at sunset",
    "width": 1024,
    "height": 1024,
    "seed": 0,
    "variant": "base",
}


def test_graph_matches_the_committed_reference():
    """Pins the emitted graph byte-for-byte against workflows/*.json.

    The reference file is what users POST directly, so it must not drift. It
    doubles as an equivalence check when this module is refactored.
    """
    built = workflow.build_workflow(workflow.resolve_params(**GOLDEN_PARAMS))
    assert built == json.loads(GOLDEN.read_text(encoding="utf-8"))


# --- LoRA adapters ----------------------------------------------------------


def test_no_lora_leaves_the_graph_untouched():
    """The feature must be inert when unused; the golden graph pins this too."""
    graph = graph_for()
    assert workflow.LORA_NODE_ID not in graph
    assert graph["guider"]["inputs"]["model"] == ["load_unet", 0]


def test_lora_is_spliced_between_the_transformer_and_the_guider():
    graph = graph_for(lora="snofs-v1.4", lora_strength=0.8)
    node = graph[workflow.LORA_NODE_ID]
    assert node["class_type"] == "LoraLoaderModelOnly"
    assert node["inputs"]["model"] == ["load_unet", 0]
    assert node["inputs"]["lora_name"] == workflow.LORAS["snofs-v1.4"].filename
    assert node["inputs"]["strength_model"] == 0.8
    # The guider must read the patched model, not the raw loader.
    assert graph["guider"]["inputs"]["model"] == [workflow.LORA_NODE_ID, 0]


def test_lora_does_not_touch_the_text_encoder_path():
    """Model-only: these adapters patch diffusion_model.* and nothing else."""
    graph = graph_for(lora="snofs-v1.4")
    assert graph["positive"]["inputs"]["clip"] == ["load_clip", 0]
    assert graph["negative"]["inputs"]["clip"] == ["load_clip", 0]


@pytest.mark.parametrize("variant", ["base", "distilled", "ponpoke-uncensored"])
def test_lora_applies_to_every_variant(variant):
    graph = graph_for(variant=variant, lora="snofs-v1.4")
    assert graph["guider"]["inputs"]["model"] == [workflow.LORA_NODE_ID, 0]
    assert graph[workflow.LORA_NODE_ID]["inputs"]["model"] == ["load_unet", 0]


def test_unknown_lora_rejected():
    with pytest.raises(workflow.WorkflowError, match="unknown lora"):
        workflow.resolve_params("x", lora="not-a-real-adapter")


def test_default_lora_strength_applied():
    params = workflow.resolve_params("x", lora="snofs-v1.4")
    assert params.lora_strength == workflow.DEFAULT_LORA_STRENGTH


def test_every_link_still_resolves_with_a_lora():
    graph = graph_for(lora="snofs-v1.4")
    for node_id, node in graph.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                assert value[0] in graph, f"{node_id}.{name} -> missing {value[0]!r}"


def test_second_adapter_is_registered_and_downloadable():
    spec = workflow.LORAS["realstockings-v2"]
    assert spec.filename == "RealStockingsV2.safetensors"
    # Trigger words are the usual reason a LoRA appears to "do nothing".
    assert spec.trigger_words == ("stockings", "RealStockings")


def test_each_adapter_gets_its_own_filename():
    """Two registry entries pointing at one file would be a copy-paste slip."""
    filenames = [spec.filename for spec in workflow.LORAS.values()]
    assert len(filenames) == len(set(filenames))


@pytest.mark.parametrize("lora", sorted(["snofs-v1.4", "realstockings-v2"]))
def test_every_registered_adapter_builds_a_valid_graph(lora):
    graph = graph_for(lora=lora)
    node = graph[workflow.LORA_NODE_ID]
    assert node["inputs"]["lora_name"] == workflow.LORAS[lora].filename
    assert graph["guider"]["inputs"]["model"] == [workflow.LORA_NODE_ID, 0]
