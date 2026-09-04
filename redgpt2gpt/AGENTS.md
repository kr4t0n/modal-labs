# AGENTS.md — RedGPT2 (Krea 2), GPT edition

Context for anyone changing this deployment. The architectural reasoning — why a
real ComfyUI runs in the container rather than a hand-written sampler — lives in
the root [`AGENTS.md`](../AGENTS.md) and is not repeated.

This file covers only what differs.

## The trap: this is not the Alternating Evaluation build

The Civitai listing is **titled** "Alternating Evaluation", and its model card
describes a scheme using two checkpoints — a high-noise model and a low-noise
model, sampled alternately in a 4H+6L configuration. Reading the card top to
bottom, it is easy to conclude this service is wired wrong because it has one
`UNETLoader`.

It is not. The card covers every edition on that page, and the AE scheme belongs
to version `3289607`, which ships two safetensors and a config JSON. The edition
deployed here, `3123514`, ships **one** file and samples conventionally.

A test asserts the graph has exactly one `UNETLoader` and that exactly one
`CivitaiFile` is fetched, with a docstring saying why. Implementing AE would
mean two loaders and an alternating sigma schedule — a different graph, and
therefore a different service directory, not an edit to this one.

## Its family is `ultra` and `finepornv4`

All three serve Krea 2 turbo as a diffusion-model-only safetensors with
Comfy-Org's Qwen3-VL-4B encoder and the Qwen-Image VAE. The graph shape is
identical across them.

`tests/test_app_config.py` asserts this service's companion files match
`ultra`'s exactly; `finepornv4` carries the same assertion against `ultra`, so
pinning both to `ultra` transitively pins all three. When changing anything
about Krea 2 handling, change all three or none.

What must **not** be shared: the app name, the Volume name, the env prefix. A
test checks this service claims its own.

## Defaults are the template's, and say so

`finepornv4` takes its sampler settings from a model card that publishes a
per-version recipe. This edition's notes cover training method and licensing and
publish **no** sampler settings, so this service falls back to ComfyUI's Krea 2
turbo template — `euler`/`simple`, 8 steps, cfg 1 — exactly as `ultra` does.

`DEFAULTS_SOURCE` says that outright and `/defaults` returns it, so a caller can
tell a template default from a published one without reading the source. A test
pins that the string admits it.

The tempting mistake is copying `finepornv4`'s `euler`/`beta` here on the
grounds that the two are both Krea 2 merges. That would present an invented
recommendation as the author's. If better settings are found, measure them and
say so in the README.

The card's "4H + 6L" is a step split for the AE edition's two-model schedule and
does not transfer to this build.

## The download is authenticated

Like `finepornv4` and unlike `ultra`, this model is NSFW-flagged and Civitai
answers `401` to an unauthenticated download — verified with a ranged GET
against the real URL, not assumed from the flag. Only `download_models` carries
the Secret; serving containers read the Volume.

The wiring is the same as `../flux2klein/app.py`, which has fetched token-gated
Civitai adapters since before this service existed: a `CIVITAI_SECRET_NAME`
constant read from `<PREFIX>_CIVITAI_SECRET` (default `civitai-secret`), passed
inline to the `download_models` decorator as
`modal.Secret.from_name(..., required_keys=["CIVITAI_TOKEN"])`. The token is
consumed in `comfyui_modal/weights.py`, which sets an `Authorization: Bearer`
header when `CIVITAI_TOKEN` is in the environment. Tests assert the constant,
following flux2klein's idiom — Modal's `Function.spec` is the only way to read
secrets back off a function and it is deprecated for removal in 1.6.0.

The failure mode without a token is not a clean error: Civitai may serve an HTML
page with a `200`, which lands in the staging file and fails the SHA256 check.
The digest turns a silent corrupt install into a refusal.

## Why the edition is in the name

Same reasoning as `finepornv4`, for a different discriminator: that listing has
several Krea 2 editions, so `redgpt2gpt` names the model plus the edition. A
deployment of the Alternating Evaluation build would be `redgpt2ae`, beside this
one rather than replacing it — existing workflows keep their endpoint URL, and
the two can be compared.

## Known gaps

- No image editing or img2img; `EmptyLatentImage` only.
- `denoise` is exposed but has no use without an input latent. It is kept
  because the shared request shape carries it and the sampler accepts it.
- Nothing verifies the deployed ComfyUI has the samplers the node offers.
  `client.py validate` covers that, against a live deployment.
- The sampler and scheduler dropdowns in the node are hand-curated and not
  checked against `workflow.py`'s defaults, unlike the finepornv4 node which has
  such a test. Worth adding if a third service grows non-template defaults.
