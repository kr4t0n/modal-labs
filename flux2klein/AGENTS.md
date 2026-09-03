# AGENTS.md — FLUX.2 klein 9B on Modal

Context for anyone changing this deployment. It is a deliberate sibling of
the other services here. The architectural reasoning — why a real ComfyUI runs
in the container rather than a hand-written sampler — lives in the root
[`AGENTS.md`](../AGENTS.md) and is not repeated.

This file covers only what differs.

## Where the code lives

Only the model-specific parts are here: the weight table and Modal object graph
(`app.py`), the request model and `/variants` route (`server.py`), the graph
(`workflow.py`) and the CLI's own arguments (`client.py`).

The container image, ComfyUI supervisor, ASGI proxy, resolution arithmetic, CLI
transport and test doubles are all in `../comfyui_modal`, shared with every other
service. The ComfyUI nodes are in `../comfy_node`.

An earlier revision copied those wholesale into each service. They were extracted
once there were two consumers, because the proxy header handling, the progress
mirror and the build-command guard all exist as fixes for bugs found the hard
way, and N copies of them lose those fixes one at a time.

## What differs from ideogram4

**One transformer, not two.** `CFGGuider` replaces `DualModelGuider`, so the
working set during sampling is ~18 GB rather than ~18.9 GB across a pair, and
peak VRAM is bounded by a single 9.4 GB checkpoint plus the 8.7 GB encoder.

**A real negative prompt.** The graph encodes it with a second `CLIPTextEncode`
rather than `ConditioningZeroOut`, so `negative_prompt` is a genuine input —
on the `base` variant. The `distilled` checkpoint is guidance-distilled and
ignores both CFG and the negative branch; the node surfaces that in its `info`
output rather than failing, because the request is still valid.

**The text encoder belongs to the variant, not the service.** `base` and
`distilled` share Comfy-Org's fp8mixed encoder; `ponpoke-uncensored` pairs the base
transformer with an abliterated bf16 one. It is deliberately not an independent
request field: an encoder is validated against a checkpoint, and letting callers
mix them arbitrarily would silently degrade output rather than fail.

ComfyUI loads it by the same path — `detect_te_model` maps a standard Qwen3-8B
state dict to `TEModel.QWEN3_8B`, which under `CLIPType.FLUX2` routes to
`klein_te(model_type="qwen3_8b")` with `KleinTokenizer8B`. A Qwen3-VL detection
lands on the same encoder for FLUX2, so the load works whether or not the
donor checkpoint kept its visual tower.

**Adapters may come from either source.** `weights.py` handles both, so an
entry is a `HuggingFaceFile` or a `CivitaiFile` depending on where it lives.
Civitai ones are token-gated and pin a digest, so `download_models` carries both
the Hugging Face and Civitai secrets. A test asserts every `CivitaiFile` has a
well-formed SHA256 — a Civitai entry without one is the failure worth catching,
since nothing upstream would notice a substituted adapter.

**Adapters are a registry, not a filename.** `lora` names an entry in `LORAS`
rather than accepting an arbitrary file, because the weights must already be on
the Volume when a request arrives — a free-form name would validate and then
fail at queue time. Each entry pairs with a `ModelFile` in `app.py`, and a test
asserts the two stay in step.

The adapter is spliced in as `LoraLoaderModelOnly` between `UNETLoader` and
`CFGGuider`, and only when one is requested: with `lora=None` the emitted graph
is byte-identical to what it was before the feature, which the committed
reference graph pins. Model-only is deliberate — these adapters patch
`diffusion_model.*` and the text encoder is a separate load here.

Note the widget order in the ComfyUI node: `lora` and `lora_strength` are
appended at the *tail* of `required`. ComfyUI matches widget values by position,
so inserting mid-list would shift every value in workflows people have saved.

**Natural-language prompts.** No structured-caption requirement, unlike Ideogram
4. There is no `/caption-template` endpoint here.

**Variants replace presets.** `variant` selects both the checkpoint *and* its
tuned `steps`/`cfg`, because those are not independent: base is 20 steps at cfg
5, distilled is 4 at cfg 1, and mixing them produces bad output rather than an
error. Both figures come from ComfyUI's official templates, not inference —
`image_flux2_text_to_image_9b.json` and
`image_flux2_klein_image_edit_9b_distilled.json`.

**The weights are gated.** The two transformers require an accepted FLUX.2
licence, so `download_models` carries a `modal.Secret` with `HF_TOKEN` and
translates `GatedRepoError` into an instruction rather than a stack trace. The
serving containers need no secret — they read the Volume.

**Explicit download destinations.** Ideogram 4's mirror happens to be laid out
exactly like ComfyUI's `models/`, so files could be downloaded in place. Here
four repos with three different internal layouts feed one directory, so every
entry in `MODEL_FILES` carries a `destination`. Downloads land in a staging
directory on the same Volume and are renamed into place — a rename, not a second
copy of 9 GB.

**A different VAE.** `full_encoder_small_decoder.safetensors`, not
`flux2-vae.safetensors`. The klein templates ship the small-decoder autoencoder;
using the flux2-dev VAE here is a silent quality regression, not an error.

## Cross-project gotcha: module names

Both services expose top-level modules named `workflow`, `server` and `app`, and
pytest collects every suite in one interpreter. Each test module therefore clears
those names from `sys.modules` before importing, and `test_workflow.py` in both
projects asserts that the module it bound actually lives in its own directory.

Without that reset the suites still *pass* — they just silently exercise
whichever project was collected first. That is why the guard test exists.
`pyproject.toml` also sets `--import-mode=importlib` so the two identically named
`test_workflow.py` files can coexist.

CI imports each `app.py` in a separate process for the same reason.

## Known gaps

- Text-to-image only. FLUX.2 klein also does image editing, and ComfyUI ships
  templates for it (`image_flux2_klein_image_edit_9b_*`); that needs
  `/upload/image` plumbing in the typed contract. The raw proxy already exposes
  the endpoint, so the graph is reachable today by posting to `/prompt` directly.
- The 4B variants are not wired up. They use a different text encoder
  (`qwen_3_4b`), which the pairing comment in `workflow.py` deliberately fixes
  to the 8B one — mixing encoder and transformer scales degrades output silently.
- Only one adapter at a time. `lora` is a single name, so the graph splices at
  most one `LoraLoaderModelOnly`; stacking would mean a chain of them and a
  strength per link.
- `trained_on` is advisory and currently defaults to `"base"` for every entry.
  Civitai's taxonomy has one bucket for klein 9B and does not distinguish the
  base transformer from the distilled one, so for Civitai-sourced adapters the
  field records an assumption rather than a published fact.
- Only one strength per adapter, applied to the whole transformer. Block-level
  or per-layer weighting would need a different loader node.

## Strength is resolved per adapter, not from a constant

`DEFAULT_LORA_STRENGTH` is a fallback, not the default. Where an author
publishes a band, `Lora.recommended_strength` carries it and
`Lora.default_strength` returns its midpoint; `resolve_params` uses that
whenever the request omits `lora_strength`.

The reason it is not a single constant: `nsfw-unlocked-v2` recommends 0.5-0.9,
*below* the old 1.0 default, so every caller taking the default was overdriving
it. Ranges are stored rather than one number because that is what model cards
say, and the spread is worth showing callers.

`None` is the only value meaning "no opinion" — an explicit `0.0` is a real
request, so the resolver tests `is not None` rather than truthiness, and a test
pins that. Correspondingly `lora_strength` is `float | None` all the way out
through the request model and the CLI flag, and the ComfyUI node grew an
`override_lora_strength` toggle rather than always sending its widget value.

**This changed behaviour for saved workflows.** A workflow saved before the
toggle existed loads with `override_lora_strength` at its `False` default, so it
now sends no strength and gets the adapter's recommendation instead of the 1.0
it used to send. That only moves the two adapters with published bands, and it
moves them toward what their authors intend, but it is a silent change on load.

## Mirrors of this registry

The ComfyUI node hardcodes both the variant and adapter name lists, because it
runs inside the user's ComfyUI where `workflow` does not exist, and because
ComfyUI calls `INPUT_TYPES()` during startup — fetching `/variants` there would
block the boot and fail whenever the endpoint is down.

`../tests/test_node_runtime.py` compares both lists against this registry. It
borrows the `workflow` module name through a context manager that restores
`sys.modules` afterwards, since every service ships one and pytest collects them
all in a single interpreter.

Without those tests, drift surfaced only at render time: the dropdown offers a
name, the server answers `unknown lora`.

## Civitai gotcha: trained words are not reliable

`trainedWords` from the Civitai API is empty for `naturalbeauty-v2`, whose model
card documents `naked` / `topless` / `bottomless` in prose, and disagrees with
the prose for `nsfw-unlocked-v2`. The registry follows whichever source is
actually correct, per-entry, with a comment saying which and why.

This matters more than it looks: an adapter registered with no triggers when it
has them loads cleanly, renders, and quietly does nothing — there is no error to
notice. Read the model card, not just the API response.
