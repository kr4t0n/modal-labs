# AGENTS.md — RedCraft v3 (Krea 2)

Context for anyone changing this deployment. The architectural reasoning — why a
real ComfyUI runs in the container rather than a hand-written sampler — lives in
the root [`AGENTS.md`](../AGENTS.md) and is not repeated.

This file covers only what differs.

## The fourth Krea 2 service

`ultra`, `finepornv4`, `redgpt2gpt` and this one all serve Krea 2 turbo as a
diffusion-model-only safetensors with Comfy-Org's Qwen3-VL-4B encoder and the
Qwen-Image VAE. The graph shape is identical across all four.

`tests/test_app_config.py` asserts this service's companion files match
`ultra`'s exactly; `finepornv4` and `redgpt2gpt` carry the same assertion, so
pinning each to `ultra` transitively pins all four. When changing anything about
Krea 2 handling, change all four or none.

What must **not** be shared: the app name, the Volume name, the env prefix. A
test checks this service claims its own.

## Two traps in the upstream listing

**The model page is a grab-bag.** Civitai model `958009` carries twenty versions
across Flux.1 D, SDXL, SD 1.5, Pony, Illustrious, Z-Image, MiniMax H3, LTX 2.5
and Krea 2. Its title mentions "LTX25 2K", which belongs to an entirely
different version. Only `3139241` is the Krea 2 build. Reading the page title
and assuming a base model would be wrong here in a way it is not elsewhere.

**Four precisions, one version id, one filename.** That single version publishes
fp8, int8, int4 and nvfp4 builds and gives all four
`redcraftHybridH3A2A_30Krea2.safetensors`. The version id identifies none of
them — only `file_id` does. This is the strongest case in the repo for pinning
the file id rather than trusting Civitai's primary-file choice, and a test
asserts both it and that the renamed destination carries the precision.

The rename is load-bearing for the same reason: `DIFFUSION_MODEL` in
`workflow.py` says `fp8` where upstream's name says nothing, and a test asserts
that too.

## Defaults are published, unlike redgpt2gpt's

The version notes read, verbatim: `ER_SDE / Euler | Simple | CFG =1 | 8-12
Steps`. So the sampler settings here are the author's, not a template fallback —
the opposite of `redgpt2gpt`, whose edition publishes nothing and therefore
takes ComfyUI's generic Krea 2 turbo values. `DEFAULTS_SOURCE` on each says
which, and `/defaults` returns it, so the two are distinguishable without
reading source. Do not sync settings between them in either direction.

`DEFAULT_SAMPLER` is `euler`, not the `er_sde` the card lists first. The card
offers them interchangeably, `euler` exists in every ComfyUI build, and a
default that failed to resolve would break every request omitting a sampler.
`ALTERNATE_SAMPLER` records the other one, `/defaults` reports it, the node
offers it, and the API accepts it. A test pins that the two differ, so a
well-meaning "fix" to match the card's ordering has to be deliberate.

Steps are 10 — the midpoint of the published 8–12 — and `STEPS_RANGE` carries
the band so `/defaults` can report it. Same convention as `finepornv4`.

## The download is anonymous, and that was verified

No `CIVITAI_TOKEN` Secret here. This listing is not NSFW-flagged and a ranged
GET against the real download URL returns `206` without credentials — checked,
not inferred from the flag.

That check matters because the reverse mistake has already been made in this
repo: `finepornv4` shipped claiming it needed no credentials, copied from
`ultra`, and its download 401s. A test asserts no secret is wired here, so the
README's claim and the code cannot drift apart.

If Civitai ever gates this file, attaching a Secret to `download_models` is the
whole fix — `comfyui_modal/weights.py` already sends the header when
`CIVITAI_TOKEN` is present. Copy the wiring from `../finepornv4/app.py`.

## Why the version is in the name

Same reasoning as `finepornv4` and `redgpt2gpt`: the upstream listing publishes
many incompatible builds, so a later RedCraft release gets a sibling directory
and its own endpoint rather than replacing this one. Existing workflows keep
their URL and the two can be compared.

The name also feeds URL derivation — see `tests/test_endpoint_convention.py`.
Directory `redcraftv3`, app `redcraftv3-comfyui`, class `RedCraftV3`, env
`REDCRAFTV3_MODAL_URL`. Break any of those and `MODAL_WORKSPACE` silently
derives a URL pointing at nothing.

## Known gaps

- No image editing or img2img; `EmptyLatentImage` only.
- `denoise` is exposed but has no use without an input latent. It is kept
  because the shared request shape carries it and the sampler accepts it.
- Nothing verifies the deployed ComfyUI has `er_sde`. `client.py validate`
  covers that, against a live deployment.
- The other three precisions are not wired up. Switching is an `app.py` edit —
  file id, digest, destination — but only one can be deployed per service.
