# AGENTS.md — ULTRA (Krea 2) on Modal

Context for anyone changing this deployment. It is a sibling of
[`../ideogram4`](../ideogram4/AGENTS.md); the architectural reasoning there —
why a real ComfyUI runs in the container rather than a hand-written sampler —
applies unchanged. This file covers what differs.

## Where the code lives

Only the model-specific parts are here: the weight table and Modal object graph
(`app.py`), the request model and `/defaults` route (`server.py`), the graph
(`workflow.py`) and the CLI's own arguments (`client.py`). Everything generic is
in `../comfyui_modal`; the ComfyUI nodes are in `../comfy_node`.

## What the model actually is

The listing calls it a checkpoint, but the file carries **only**
`model.diffusion_model.*` — 878 tensors, no CLIP, no VAE, with `weight_scale`
tensors marking it int8. So it loads through `UNETLoader`, not
`CheckpointLoaderSimple`, and the encoder and VAE are separate loads. ComfyUI
strips the `model.diffusion_model.` prefix itself via
`unet_prefix_from_state_dict`, so no key conversion is needed here.

**The companions are not interchangeable with the klein services'.** Krea 2 taps
a **4B** Qwen3-VL across 12 layers (`CLIPType.KREA2`) and uses the **Qwen-Image**
autoencoder. Substituting the 8B encoder or a Flux-family VAE degrades output
rather than failing loudly.

## Provenance, and why the digest matters here

This model was originally handed over as a `civitai.red` link — a clone of the
Civitai frontend, serving the same UI and proxying the same API routes, on a
certificate issued days earlier. The model itself is genuine and on
`civitai.com`, so the fetch points there.

The `CivitaiFile` fetcher verifies the published SHA256 before installing and
refuses on mismatch. That is the property worth preserving: it makes a
substituted file fail closed regardless of which host serves the bytes, so
pointing at a mirror for access reasons stays safe as long as the digest stays
pinned. Do not remove the digest to "simplify" the fetch.

`.safetensors` also cannot execute code on load the way a pickled `.ckpt` can,
so the realistic risk was substituted weights rather than RCE — but the digest
closes that too.

## Sampler defaults, and the open question

Defaults come from ComfyUI's `image_krea2_turbo_t2i_int8` template: 8 steps,
cfg 1, `euler`/`simple`. Those are verified, not inferred.

**Whether ULTRA descends from Krea 2 turbo or raw is unknown.** Its listing
publishes no settings, no trained words, and a one-line description; both
branches are the same size in int8 so the file cannot settle it. The turbo
defaults are the conservative choice — cfg 1 on a raw model looks undercooked,
which is recoverable, whereas cfg 5 on a distilled model looks broken. If
someone establishes the lineage, record it here.

The negative branch mirrors the template: `ConditioningZeroOut` by default,
because at cfg 1 it is never consulted. Supplying `negative_prompt` swaps in a
real `CLIPTextEncode` under the same node id, so the sampler's wiring does not
change either way — a test covers both shapes.

## Known gaps

- Text-to-image only. The Krea 2 templates also cover image style reference,
  which needs `/upload/image` plumbing.
- No LoRA support, though `Comfy-Org/Krea-2` ships eleven style adapters and
  `flux2klein` already demonstrates the registry pattern. Whether those adapters
  behave on a finetune rather than stock Krea 2 is untested.
- The checkpoint is pinned by Civitai `model_version_id`. Bumping to a future
  version means updating the id, filename and SHA256 together; the golden-graph
  test fails until the filename in `workflow.py` matches.
