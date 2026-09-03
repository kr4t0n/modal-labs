# AGENTS.md — FinePorn v4 (Krea 2)

Context for anyone changing this deployment. The architectural reasoning — why a
real ComfyUI runs in the container rather than a hand-written sampler — lives in
the root [`AGENTS.md`](../AGENTS.md) and is not repeated.

This file covers only what differs.

## Why the version is in the name

Every other service here is named for its model (`ultra`, `flux2klein`) and
tracks new releases in place. This one carries `v4` in the directory, the Modal
app, the Volume, the env prefix and the node id, so that a future FinePorn
release can be deployed **beside** it rather than replacing it.

That is a deliberate trade. Upgrading in place would mean one endpoint URL and
one Volume, but every saved ComfyUI workflow and every client would switch
models the moment the deploy landed, with no way to compare the two or roll
back without a redeploy. Side-by-side costs a second ~31 GB Volume and a second
idle deployment, and buys A/B comparison and an unchanged URL for existing work.

So a `finepornv5/` is a **new directory**, not an edit to this one. What it can
share is already shared: the Krea 2 companions come from the same Hugging Face
repo, and everything generic lives in `comfyui_modal`. What it must not share is
the Volume name or the app name.

The weight file keeps the name `fineporn_v4_bf16.safetensors` — already
version-distinct, and each service has its own Volume, so it needs no prefix.

## Its closest relative is `ultra`, not `flux2klein`

Both this and `../ultra` serve **the same base model**: Krea 2 turbo, as a
diffusion-model-only safetensors, with Comfy-Org's Qwen3-VL-4B encoder and the
Qwen-Image VAE loaded beside it. The graph shape is identical — `UNETLoader` +
`CLIPLoader` + `KSampler`, with `ConditioningZeroOut` on the negative branch.

A test in `tests/test_app_config.py` asserts the two services request byte-identical
companion files. If they ever diverge it means one picked up a different encoder
or VAE build, which changes output without failing.

When changing anything about Krea 2 handling, change both or neither.

## What differs from ultra

**Defaults come from the model card, not a ComfyUI template.** ULTRA's listing
publishes no sampler settings, so that service follows ComfyUI's generic Krea 2
turbo template (`euler`/`simple`, 8 steps). This card publishes settings per
version, and its V4 section names `euler + beta` and `er_sde + simple`. Hence
`DEFAULT_SCHEDULER = "beta"` here — the single most likely thing to get
"corrected" back to `simple` by someone assuming the two services should match.
A test pins it with that reasoning in the docstring.

Steps are 10 because the card gives 8–10 for v1 and 8–12 for v2 and restates no
figure for v4; 10 is the midpoint of the published band. `STEPS_RANGE` carries
the band so `/defaults` can report it.

**It renders above 1 MP by default.** This is the only service here that does.
The card reports standard Krea 2 resolutions underperform on this merge and
publishes a scaling table, so `DEFAULT_SIDE = 1280` and
`DEFAULT_MEGAPIXELS = 1.64` (the card's own ×1.25 of 1024²).

That default has to be set in **three** places, because two of them always send
a value and would otherwise silently override the server:

- `server.GenerateRequest` redeclares `width`, `height` and `megapixels`.
  Redeclaring a Pydantic field replaces it wholesale, so the bounds are restated
  rather than inherited.
- `client.py` passes `default_side` / `default_megapixels` to
  `cli.add_geometry_arguments`, because `cli.geometry_payload` always puts
  width and height in the payload.
- `comfy_node/nodes_finepornv4.py` passes the same to `common_geometry_inputs`,
  because ComfyUI widgets always send their value.

Both shared helpers grew keyword-only parameters defaulting to the old 1024/1.0,
so no other service changed. Tests assert that: one pins this node's defaults
against `workflow`, another pins every *other* node at 1024/1.0.

**It is a merge, not a finetune.** The card credits a list of third-party LoRAs
baked in at balanced weights. That matters for licence questions — terms follow
from the components — and it explains why there are no trained words: nothing
was trained, so there is no token to invoke.

**Prompt guidance is reported, never injected.** The card asks for a
smartphone-snapshot opener and the merge renders flatter without one, so
`PROMPT_GUIDANCE` carries the advice and `/defaults` surfaces it. Prepending it
server-side was deliberately rejected: `params.prompt` in the response is
supposed to be what was actually sent, and silently rewriting it would make that
a lie. A test asserts the graph carries the caller's prompt verbatim.

## The four precisions

The upstream listing publishes int8, nvfp4, fp8 and bf16 under one model id, and
**the int8 and fp8 builds share a version id and a filename**. So the
`CivitaiFile` pins the file id as well as the version, and a test asserts both
plus `bf16` appearing in the destination.

The upstream filename — `finepornV4INT8NVFP4BF16_v4Bf16.safetensors` — names all
four precisions for a file that is only one of them, so the destination renames
it. `DIFFUSION_MODEL` in `workflow.py` is the renamed form.

bf16 is what this deployment was asked for; it is also the largest at 25.7 GB and
gives the slowest cold start of any service in the repo. README documents the
alternatives and what switching involves.

## Known gaps

- No image editing or img2img; `EmptyLatentImage` only.
- `denoise` is exposed but has no use without an input latent. It is kept
  because the shared request shape carries it and the sampler accepts it.
- The sampler list in the ComfyUI node is hand-curated. `res_2s` is named on the
  card for v1 but omitted, being absent from some ComfyUI builds; the typed API
  accepts any sampler string, so it stays reachable there.
- Nothing verifies the deployed ComfyUI actually has `beta` and `er_sde`. That
  is what `client.py validate` is for, and it needs a live deployment.
