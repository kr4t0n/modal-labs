# AGENTS.md — WAI-illustrious-SDXL on Modal

Context for anyone changing this deployment. It is a sibling of
[`../ideogram4`](../ideogram4/AGENTS.md); the architectural reasoning there —
why a real ComfyUI runs in the container rather than a hand-written sampler —
applies unchanged and is not repeated. This file covers what differs.

## Where the code lives

Only the model-specific parts are here: the Civitai fetch and Modal object graph
(`app.py`), the request model and `/defaults` route (`server.py`), the graph
(`workflow.py`) and the CLI's own arguments (`client.py`). Everything generic is
in `../comfyui_modal`; the ComfyUI nodes are in `../comfy_node`.

## What differs from the other services

**A single-file checkpoint, not split weights.** The other two load separate
UNet, text-encoder and VAE files. This is one ~6.8 GB fp16 SDXL `.safetensors`
in the original CompVis layout (`conditioner.embedders.*`), so it loads through
`CheckpointLoaderSimple`, whose three outputs are MODEL, CLIP and VAE. That is
why `decode` reads `["load_checkpoint", 2]` rather than a `VAELoader`, and why
`extra_model_paths.yaml` maps `checkpoints` rather than three folders.

**A plain `KSampler`, not a custom-sampler chain.** No guider, no scheduler node,
no separate sigmas — SDXL predates all of that. `sampler_name` *and* `scheduler`
are both request fields here; the other services only expose `sampler_name`
because their schedule comes from a dedicated node.

**Clip skip is load-bearing.** Booru-tagged SDXL finetunes are trained against
the penultimate CLIP layer. `CLIPSetLastLayer(-2)` sits between the checkpoint's
CLIP output and *both* text encoders. Wiring it to only one would silently skew
guidance rather than fail, so a test asserts both read the skipped CLIP.

**Sampler defaults are conventions, not published values.** Civitai's API
returns an empty description and no `trainedWords` for this model, so the
defaults (28 steps, cfg 5, `euler_ancestral`/`normal`, clip skip -2, the standard
Danbooru negative) come from community practice. They are documented as such and
every one is overridable. Do not present them as author recommendations.

**Civitai, not Hugging Face.** No credentials are needed — verified by fetching
the file's first bytes anonymously and getting a valid safetensors header back.
An earlier 403 on `HEAD` was the presigned URL being signed for GET only, not an
auth wall.

Two consequences for `download_models`:

- The API URL answers with a **24-hour presigned CDN link**, so the redirect must
  be followed at download time rather than a URL pinned in code.
- There is no HF-style integrity guarantee, so the fetch **streams and verifies
  Civitai's published SHA256** before installing, and refuses to install on a
  mismatch. This is the only check standing between a third-party CDN and the
  serving container.

`CIVITAI_TOKEN` is read from the environment if present. Nothing sets it today;
it exists so that if Civitai ever gates downloads, attaching a Modal Secret is
the entire fix, with no code change.

**The GPU default is A10, not H100.** ~7 GB working set, and SDXL is fp16
throughout — so the reason the other services avoid Ampere (no fp8 tensor cores)
does not apply here. A test pins the default, since defaulting higher silently
costs about four times as much per hour.

## Known gaps

- Text-to-image only. No img2img, despite `denoise` being plumbed through — that
  would need `/upload/image` and a `VAEEncode` branch.
- No LoRA loading, which is the main thing users of this model reach for. It
  would mean a `LoraLoader` chain and a way to ship LoRA files into the Volume.
- No refiner or hires-fix pass.
- The checkpoint version is pinned by Civitai `model_version_id`. Bumping to a
  future v18 means updating the id, the filename and the SHA256 together — the
  golden-graph test will fail until the filename in `workflow.py` matches.
