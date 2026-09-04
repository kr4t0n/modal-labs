# AGENTS.md — Dark Beast v3 (Krea 2)

Context for anyone changing this deployment. The architectural reasoning — why a
real ComfyUI runs in the container rather than a hand-written sampler — lives in
the root [`AGENTS.md`](../AGENTS.md) and is not repeated.

This file covers only what differs.

## The fifth Krea 2 service

`ultra`, `finepornv4`, `redgpt2gpt`, `redcraftv3` and this one all serve Krea 2
turbo as a diffusion-model-only safetensors with Comfy-Org's Qwen3-VL-4B encoder
and the Qwen-Image VAE. The graph shape is identical across all five.

`tests/test_app_config.py` asserts this service's companion files match
`ultra`'s exactly; the other three carry the same assertion, so pinning each to
`ultra` transitively pins all five. When changing anything about Krea 2
handling, change all five or none.

What must **not** be shared: the app name, the Volume name, the env prefix. A
test checks this service claims its own.

## The listing describes a video model. This is not it.

Civitai model `2242173` is titled "H3 Director Edition" and its description is
entirely about a video pipeline — automated short-drama production, 2K
single-pass sampling, VSR and RIFE frame interpolation, timings quoted in
seconds-per-8-second-clip. Read top to bottom, it looks nothing like a
text-to-image service.

That product is version `3274224`, whose base model is **MiniMax H3**. The
version deployed here, `3173268`, is **Krea 2** and produces stills. The listing
also carries FLUX.2 klein, Z-Image Turbo and SDXL versions.

Two consequences worth guarding, and both are tested:

- The graph is a still-image graph — one `EmptyLatentImage`, one `KSampler`, one
  `SaveImage`. A test asserts that shape, so anything resembling the H3 pipeline
  appearing here means the wrong base model was wired up.
- The description's **"6-10 steps"** belongs to H3. It is not this model's
  recommendation, and `DEFAULT_STEPS` is 8 from ComfyUI's Krea 2 turbo template
  instead. This is the same trap `redgpt2gpt` has with "4H + 6L" — a number
  sitting in prose next to a model it does not describe.

Contrast `redcraftv3`, by the same author over the same base, whose version
notes *do* publish a recipe. `DEFAULTS_SOURCE` on each says which it is and
`/defaults` returns it, so the two are distinguishable without reading source.
Do not sync settings between them in either direction.

## Five precisions, one version id, one filename

The deployed version publishes int8, fp8, bf16, nvfp4 and int4 builds, all named
`darkBeastH3Director_darkBeast330.safetensors`. The version id identifies none
of them — only `file_id` does. The int8 build is pinned, which is also the
version's primary file.

The rename is load-bearing twice over: `DIFFUSION_MODEL` says both `int8` and
`krea2` where upstream's name says neither, and a test asserts both substrings
appear. On a listing that spans five base models, "which base is this file"
matters as much as "which precision".

## The NSFW flag predicts nothing

No `CIVITAI_TOKEN` Secret here. Verified with a ranged GET against the real
download URL: `206`, with the response body's first bytes decoding as a
safetensors header rather than an HTML error page.

That check is worth keeping in mind because this model **is** NSFW-flagged and
still downloads anonymously, while `finepornv4` and `redgpt2gpt` are flagged and
`401`. Gating is a per-model setting on Civitai's side; the flag does not decide
it. `comfyui_modal/weights.py` says so at the point the token is used, and a
test here asserts no secret is wired so the README's claim cannot drift from the
code.

Checking a status code alone is not enough either: an unauthenticated Civitai
request can answer `200` with an HTML error page. Look at the bytes.

If this file is ever gated, attaching a Secret to `download_models` is the whole
fix — copy the wiring from `../finepornv4/app.py`.

## Why the version is in the name

Same reasoning as `finepornv4`, `redgpt2gpt` and `redcraftv3`: the upstream
listing publishes many incompatible builds, so a later Dark Beast release gets a
sibling directory and its own endpoint rather than replacing this one.

The name also feeds URL derivation — see `tests/test_endpoint_convention.py`.
Directory `darkbeastv3`, app `darkbeastv3-comfyui`, class `DarkBeastV3`, env
`DARKBEASTV3_MODAL_URL`. Break any of those and `MODAL_WORKSPACE` silently
derives a URL pointing at nothing.

## Known gaps

- No image editing or img2img; `EmptyLatentImage` only.
- `denoise` is exposed but has no use without an input latent. It is kept
  because the shared request shape carries it and the sampler accepts it.
- The H3 video edition is not wired up and would not fit this shape: it needs a
  video graph, a different base model and a different container. That is a new
  service, not a config change.
- The other four precisions are not wired up. Switching is an `app.py` edit —
  file id, digest, destination — but only one can be deployed per service.
