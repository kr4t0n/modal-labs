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
    """An adapter with no published range still gets the service default."""
    params = workflow.resolve_params("x", lora="snofs-v1.4")
    assert workflow.LORAS["snofs-v1.4"].recommended_strength is None
    assert params.lora_strength == workflow.DEFAULT_LORA_STRENGTH


@pytest.mark.parametrize(
    ("lora", "expected"),
    [("nsfw-unlocked-v2", 0.7), ("realism-engine-v2", 1.125), ("snofs-v1.4", 1.0)],
)
def test_omitted_strength_resolves_from_the_adapter(lora, expected):
    """The point of the field: 1.0 overdrives an adapter whose band is 0.5-0.9."""
    assert workflow.resolve_params("x", lora=lora).lora_strength == expected


@pytest.mark.parametrize("explicit", [0.0, 0.35, 1.0, 2.0])
def test_explicit_strength_always_wins(explicit):
    """Including an explicit 1.0, and including a falsy 0.0.

    `None` is the only "no opinion" value; testing 0.0 pins that the resolver
    checks for None rather than truthiness.
    """
    params = workflow.resolve_params("x", lora="nsfw-unlocked-v2", lora_strength=explicit)
    assert params.lora_strength == explicit


def test_published_strength_ranges_are_well_formed():
    for name, spec in workflow.LORAS.items():
        if spec.recommended_strength is None:
            continue
        low, high = spec.recommended_strength
        assert 0.0 < low <= high, name
        assert low <= spec.default_strength <= high, name


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


def test_third_adapter_is_registered_and_downloadable():
    spec = workflow.LORAS["realism-engine-v2"]
    assert spec.filename == "Realism_Engine_Klein_V2.safetensors"
    # Upstream publishes no trained words: this is a general finetune, not a
    # concept adapter, so an empty tuple is the correct registry value.
    assert spec.trigger_words == ()


@pytest.mark.parametrize(
    ("name", "filename", "triggers"),
    [
        (
            "nsfw-unlocked-v2",
            "FLUX2_KLEIN_UNLOCKED_V2.safetensors",
            ("nude", "naked", "blow job", "cum", "ass", "pussy"),
        ),
        (
            "naturalbeauty-v2",
            "NaturalBeautyFLUX2Klein9BNudity_v2.safetensors",
            ("naked", "topless", "bottomless"),
        ),
    ],
)
def test_multi_architecture_adapters_pin_their_klein_build(name, filename, triggers):
    """Both upstreams publish builds for other architectures under one model.

    The filename is what proves the klein build was the one registered. The
    triggers matter twice over for NaturalBeauty, whose machine-readable list
    is empty upstream while its model card documents them in prose.
    """
    spec = workflow.LORAS[name]
    assert spec.filename == filename
    assert spec.trigger_words == triggers


def test_each_adapter_gets_its_own_filename():
    """Two registry entries pointing at one file would be a copy-paste slip."""
    filenames = [spec.filename for spec in workflow.LORAS.values()]
    assert len(filenames) == len(set(filenames))


# Derived from the registry rather than hardcoded: the literal list this
# replaces went stale twice, once per adapter added.
@pytest.mark.parametrize("lora", sorted(workflow.LORAS))
def test_every_registered_adapter_builds_a_valid_graph(lora):
    graph = graph_for(lora=lora)
    node = graph[workflow.LORA_NODE_ID]
    assert node["inputs"]["lora_name"] == workflow.LORAS[lora].filename
    assert graph["guider"]["inputs"]["model"] == [workflow.LORA_NODE_ID, 0]


# --- Image edit -------------------------------------------------------------
# Flattened from ComfyUI's `image_flux2_klein_image_edit_9b_base`. The graph
# below is the same text-to-image one plus a reference chain, so the tests that
# matter are the ones pinning what the template does differently.


def edit_graph(names=("ref_0.png",), **kwargs):
    return workflow.build_workflow(
        workflow.resolve_params("edit this", reference_images=names, **kwargs)
    )


def test_no_references_leaves_the_graph_untouched():
    """The whole feature must be invisible to text-to-image callers."""
    graph = graph_for()
    assert workflow.REFERENCE_ENCODE_NODE_ID not in graph
    assert not any(n["class_type"] == "ReferenceLatent" for n in graph.values())
    assert graph["guider"]["inputs"]["positive"] == ["positive", 0]
    assert graph["guider"]["inputs"]["negative"] == ["negative", 0]


def test_reference_is_loaded_scaled_and_encoded():
    graph = edit_graph()
    assert graph[f"{workflow.REFERENCE_SCALE_NODE_ID}_0"]["inputs"]["image"] == "ref_0.png"
    scale = graph[workflow.REFERENCE_SCALE_NODE_ID]
    assert scale["class_type"] == "ImageScaleToTotalPixels"
    assert scale["inputs"]["upscale_method"] == "lanczos"
    assert scale["inputs"]["megapixels"] == 1.0
    encode = graph[workflow.REFERENCE_ENCODE_NODE_ID]
    assert encode["class_type"] == "VAEEncode"
    # Encoded from the *scaled* image, and against the graph's own VAE.
    assert encode["inputs"]["pixels"] == [workflow.REFERENCE_SCALE_NODE_ID, 0]
    assert encode["inputs"]["vae"] == ["load_vae", 0]


def test_both_branches_receive_a_reference():
    """The template wires ReferenceLatent into the negative branch too.

    Wiring only the positive still renders, so nothing but this test would
    notice the drift from the published recipe.
    """
    graph = edit_graph()
    for branch, node_id in (
        ("positive", workflow.REFERENCE_POSITIVE_NODE_ID),
        ("negative", workflow.REFERENCE_NEGATIVE_NODE_ID),
    ):
        node = graph[node_id]
        assert node["class_type"] == "ReferenceLatent"
        assert node["inputs"]["conditioning"] == [branch, 0]
        assert node["inputs"]["latent"] == [workflow.REFERENCE_ENCODE_NODE_ID, 0]
        assert graph["guider"]["inputs"][branch] == [node_id, 0]


def test_the_reference_sets_the_output_size():
    """`GetImageSize` on the scaled reference drives both size consumers."""
    graph = edit_graph(width=512, height=512)
    for node_id in ("sigmas", "latent"):
        assert graph[node_id]["inputs"]["width"] == [workflow.REFERENCE_SIZE_NODE_ID, 0]
        assert graph[node_id]["inputs"]["height"] == [workflow.REFERENCE_SIZE_NODE_ID, 1]
    assert graph[workflow.REFERENCE_SIZE_NODE_ID]["inputs"]["image"] == [
        workflow.REFERENCE_SCALE_NODE_ID,
        0,
    ]


def test_params_do_not_claim_a_size_they_did_not_apply():
    """A caller reading params must not think width/height took effect."""
    params = workflow.resolve_params("x", reference_images=("a.png",), width=512, height=512)
    reported = params.as_dict()
    assert reported["width"] is None and reported["height"] is None
    assert reported["is_edit"] is True
    assert reported["reference_images"] == ["a.png"]
    # Text-to-image keeps reporting real numbers.
    plain = workflow.resolve_params("x", width=512, height=512).as_dict()
    assert plain["width"] == 512 and plain["is_edit"] is False


def test_multiple_references_chain_per_branch():
    graph = edit_graph(names=("a.png", "b.png", "c.png"))
    encodes = [n for n, v in graph.items() if v["class_type"] == "VAEEncode"]
    assert len(encodes) == 3
    # Each branch chains one ReferenceLatent per reference, in series.
    for node_id in (workflow.REFERENCE_POSITIVE_NODE_ID, workflow.REFERENCE_NEGATIVE_NODE_ID):
        chain = [n for n in graph if n.startswith(node_id)]
        assert len(chain) == 3
    refs = [n for n, v in graph.items() if v["class_type"] == "ReferenceLatent"]
    assert len(refs) == 6


def test_only_the_first_reference_defines_the_geometry():
    """Later references are conditioning only; they must not resize anything."""
    graph = edit_graph(names=("a.png", "b.png"))
    assert graph[workflow.REFERENCE_SIZE_NODE_ID]["inputs"]["image"] == [
        workflow.REFERENCE_SCALE_NODE_ID,
        0,
    ]
    assert sum(v["class_type"] == "GetImageSize" for v in graph.values()) == 1


def test_reference_megapixels_reaches_every_scaler():
    graph = edit_graph(names=("a.png", "b.png"), reference_megapixels=2.5)
    scalers = [v for v in graph.values() if v["class_type"] == "ImageScaleToTotalPixels"]
    assert scalers and all(s["inputs"]["megapixels"] == 2.5 for s in scalers)


def test_editing_composes_with_a_lora():
    graph = edit_graph(lora="snofs-v1.4")
    assert graph["guider"]["inputs"]["model"] == [workflow.LORA_NODE_ID, 0]
    assert graph["guider"]["inputs"]["positive"] == [workflow.REFERENCE_POSITIVE_NODE_ID, 0]


def test_every_link_resolves_when_editing():
    graph = edit_graph(names=("a.png", "b.png"))
    for node_id, node in graph.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                assert value[0] in graph, f"{node_id}.{name} -> missing {value[0]!r}"


def test_the_distilled_variant_refuses_to_edit():
    """No upstream edit template exists for it; it would render something else."""
    with pytest.raises(workflow.WorkflowError, match="distilled"):
        workflow.resolve_params("x", variant="distilled", reference_images=("a.png",))


def test_editing_refuses_a_batch():
    """The reference fixes one output size, so a batch is N copies of one edit."""
    with pytest.raises(workflow.WorkflowError, match="batch_size"):
        workflow.resolve_params("x", reference_images=("a.png",), batch_size=2)


def test_uncensored_variant_may_edit():
    """It is the base transformer with a different encoder, so the recipe holds."""
    graph = edit_graph(variant="ponpoke-uncensored")
    assert graph["load_unet"]["inputs"]["unet_name"] == "flux-2-klein-base-9b-fp8.safetensors"
    assert graph["guider"]["inputs"]["positive"] == [workflow.REFERENCE_POSITIVE_NODE_ID, 0]
