"""Graph fragments that more than one service's `build_workflow` needs.

A service owns its model's graph; this holds only the pieces that are genuinely
architecture-independent, so six near-identical copies do not drift apart.
"""

from __future__ import annotations

from typing import Any

# Node ids for the img2img source chain. Fixed rather than caller-supplied so
# tests and the committed reference graphs can name them.
SOURCE_LOAD_NODE_ID = "source_load"
SOURCE_SCALE_NODE_ID = "source_scale"
SOURCE_ENCODE_NODE_ID = "source_encode"

# Matches the image-edit path in flux2klein, which takes it from ComfyUI's own
# klein edit template. One resampling method across the repo is one fewer thing
# to explain, and nothing suggests another.
SOURCE_UPSCALE_METHOD = "lanczos"


def splice_img2img_source(
    graph: dict[str, Any],
    *,
    filename: str,
    megapixels: float,
    sampler_node_id: str = "sample",
    latent_node_id: str = "latent",
    vae_node_id: str = "load_vae",
) -> None:
    """Sample from an encoded image instead of an empty latent, in place.

    Plain ComfyUI img2img: load the file, normalise it to a pixel budget, encode
    it, and hand that to the sampler as its starting latent. How far it is then
    re-noised is `denoise` on the sampler, which is already a request field on
    every service this is used from.

    Two things worth knowing:

    * **The encoded image defines the latent**, so the output takes its size
      from the source and whatever width/height the caller asked for no longer
      applies. The empty-latent node is therefore removed rather than left
      orphaned — ComfyUI would skip it, but a dead node in an API graph that
      users read and POST is just a lie about what runs.
    * **No step compensation is needed.** `comfy.samplers.KSampler.set_steps`
      builds a `steps / denoise` schedule and keeps the last `steps + 1` sigmas,
      so a turbo model asked for 8 steps at denoise 0.5 still samples 8 times.
      Scaling steps here would double-count that.
    """
    graph[SOURCE_LOAD_NODE_ID] = {
        "class_type": "LoadImage",
        "inputs": {"image": filename},
        "_meta": {"title": "Source image"},
    }
    graph[SOURCE_SCALE_NODE_ID] = {
        "class_type": "ImageScaleToTotalPixels",
        "inputs": {
            "image": [SOURCE_LOAD_NODE_ID, 0],
            "upscale_method": SOURCE_UPSCALE_METHOD,
            "megapixels": megapixels,
            "resolution_steps": 1,
        },
        "_meta": {"title": "Normalise source"},
    }
    graph[SOURCE_ENCODE_NODE_ID] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": [SOURCE_SCALE_NODE_ID, 0], "vae": [vae_node_id, 0]},
        "_meta": {"title": "Encode source"},
    }

    graph[sampler_node_id]["inputs"]["latent_image"] = [SOURCE_ENCODE_NODE_ID, 0]
    graph.pop(latent_node_id, None)
