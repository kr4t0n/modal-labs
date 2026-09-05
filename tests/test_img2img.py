"""img2img across every service that supports it.

The splice is shared code (`comfyui_modal.graph`), so its invariants are
asserted once here against every service rather than copied into six suites
that would drift.

`flux2klein` is deliberately absent: it samples through `SamplerCustomAdvanced`
where `denoise` is not automatic, and it already edits images by conditioning
from a reference. Different mechanism, different tests.
"""

from __future__ import annotations

import json

import pytest

from comfyui_modal import graph as graph_fragments
from comfyui_modal.testing import service_workflow

# Every service whose sampler is the `KSampler` node, which is what makes
# `denoise` work without a step-compensation hack.
SERVICES = [
    "ultra",
    "finepornv4",
    "redgpt2gpt",
    "redcraft3",
    "darkbeast3",
    "zimageturbostableyogi",
]


def img2img_graph(workflow, **kwargs):
    params = workflow.resolve_params("a test prompt", source_image="src.png", **kwargs)
    return workflow.build_workflow(params), params


@pytest.mark.parametrize("service", SERVICES)
def test_text_to_image_is_untouched(service):
    """The feature must be invisible unless a source is supplied.

    Each service also pins its text-to-image graph byte-for-byte against a
    committed reference; this is the cheaper statement of the same thing.
    """
    with service_workflow(service) as workflow:
        graph = workflow.build_workflow(workflow.resolve_params("x"))
    assert graph_fragments.SOURCE_ENCODE_NODE_ID not in graph
    assert "latent" in graph, "the empty latent must survive when not editing"
    assert graph["sample"]["inputs"]["latent_image"] == ["latent", 0]


@pytest.mark.parametrize("service", SERVICES)
def test_the_sampler_starts_from_the_encoded_source(service):
    with service_workflow(service) as workflow:
        graph, _ = img2img_graph(workflow)

    assert graph[graph_fragments.SOURCE_LOAD_NODE_ID]["inputs"]["image"] == "src.png"
    scale = graph[graph_fragments.SOURCE_SCALE_NODE_ID]
    assert scale["class_type"] == "ImageScaleToTotalPixels"
    encode = graph[graph_fragments.SOURCE_ENCODE_NODE_ID]
    assert encode["class_type"] == "VAEEncode"
    # Encoded from the *scaled* image, against the graph's own VAE.
    assert encode["inputs"]["pixels"] == [graph_fragments.SOURCE_SCALE_NODE_ID, 0]
    assert encode["inputs"]["vae"] == ["load_vae", 0]
    assert graph["sample"]["inputs"]["latent_image"] == [graph_fragments.SOURCE_ENCODE_NODE_ID, 0]


@pytest.mark.parametrize("service", SERVICES)
def test_the_empty_latent_is_removed_not_orphaned(service):
    """ComfyUI would skip a dangling node, but users read and POST these graphs.

    Leaving it in would advertise a width/height that never runs.
    """
    with service_workflow(service) as workflow:
        graph, _ = img2img_graph(workflow)
    assert "latent" not in graph
    assert not any(n["class_type"].startswith("Empty") for n in graph.values())


@pytest.mark.parametrize("service", SERVICES)
def test_denoise_reaches_the_sampler(service):
    """The whole point: `denoise` was inert until there was a source latent."""
    with service_workflow(service) as workflow:
        graph, _ = img2img_graph(workflow, denoise=0.55)
    assert graph["sample"]["inputs"]["denoise"] == 0.55


@pytest.mark.parametrize("service", SERVICES)
def test_params_do_not_claim_a_size_they_did_not_apply(service):
    with service_workflow(service) as workflow:
        _, params = img2img_graph(workflow, width=512, height=512)
        plain = workflow.resolve_params("x", width=512, height=512).as_dict()

    reported = params.as_dict()
    assert reported["is_img2img"] is True
    assert reported["width"] is None and reported["height"] is None
    # Text-to-image keeps reporting real numbers.
    assert plain["width"] == 512 and plain["is_img2img"] is False


@pytest.mark.parametrize("service", SERVICES)
def test_source_megapixels_reaches_the_scaler(service):
    with service_workflow(service) as workflow:
        graph, _ = img2img_graph(workflow, source_megapixels=2.5)
    assert graph[graph_fragments.SOURCE_SCALE_NODE_ID]["inputs"]["megapixels"] == 2.5


@pytest.mark.parametrize("service", SERVICES)
def test_a_batch_is_refused(service):
    """One encoded source is one latent, so a batch is N copies of one render."""
    # Bound inside the context, called outside it: the module object stays
    # alive through these references even once sys.modules is restored.
    with service_workflow(service) as workflow:
        error, resolve = workflow.WorkflowError, workflow.resolve_params
    with pytest.raises(error, match="batch_size"):
        resolve("x", source_image="a.png", batch_size=2)


@pytest.mark.parametrize("service", SERVICES)
def test_an_absurd_source_budget_is_refused(service):
    with service_workflow(service) as workflow:
        error, resolve = workflow.WorkflowError, workflow.resolve_params
    with pytest.raises(error, match="source_megapixels"):
        resolve("x", source_image="a.png", source_megapixels=99.0)


@pytest.mark.parametrize("service", SERVICES)
def test_every_link_resolves_and_the_graph_serialises(service):
    with service_workflow(service) as workflow:
        graph, _ = img2img_graph(workflow)
    json.dumps(graph)
    for node_id, node in graph.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and value and isinstance(value[0], str):
                assert value[0] in graph, f"{service}: {node_id}.{name} -> missing {value[0]!r}"


@pytest.mark.parametrize("service", SERVICES)
def test_the_service_declares_the_upload_field(service):
    """Without this the base64 never reaches ComfyUI and LoadImage names a lie."""
    with service_workflow(service):
        import server

        assert "source_image" in server.SERVICE.upload_fields
