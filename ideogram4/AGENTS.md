# AGENTS.md — Ideogram 4 on Modal

Context for anyone (human or agent) changing this deployment.

## The core decision

Ideogram 4 has open weights *and* first-class ComfyUI core support. There were
two ways to serve it:

1. Load the weights with `diffusers` and write a sampling loop.
2. Run a real headless ComfyUI on the GPU and put an HTTP layer in front.

This project does (2). The model's logit-normal noise schedule
(`Ideogram4Scheduler`), dual-branch classifier-free guidance (`DualModelGuider`,
two separate 9.3 GB transformers) and the fp8 kernels are all upstream code
that gets fixed and tuned by people who work on it full time. Reimplementing
them buys nothing and silently drifts. The cost is a heavier container and a
process to supervise.

It also answers the actual requirement directly: because ComfyUI *is* the
server, the deployed URL speaks the ComfyUI protocol with no translation layer,
so any ComfyUI client works against it unmodified.

## Components

| File | Responsibility |
| --- | --- |
| `app.py` | Modal wiring only: container image, weights Volume, GPU class, the two web endpoints, the `modal run` entrypoint. No request logic. |
| `server.py` | The ASGI app. Typed `/generate` contract plus a transparent reverse proxy for everything else. Runs only inside the container. |
| `workflow.py` | Pure function: parameters → ComfyUI API-format graph. No I/O, no Modal, no HTTP. This is what the tests exercise. |
| `client.py` | Local CLI. Also holds `validate`, which is the schema-drift check. |
| `comfy_node/` | Runs inside the *user's* ComfyUI, not this project's environment. Depends only on what ComfyUI already ships (torch, numpy, PIL, requests). |

The split matters: `workflow.py` is the only place that knows the graph, so a
ComfyUI upgrade that renames a node input is a one-file change, and
`client.py validate` proves it against a live deployment.

## How the graph was derived

`workflow.py` is a flattened form of ComfyUI's official `image_ideogram4_t2i`
template. The template drives the sampler through utility nodes — a preset
lookup table (`JsonExtractString` → `StringReplace` → `ComfyNumberConvert`), a
`CustomCombo` for preset choice, `ComfyMathExpression` for dimension snapping.
None of those touch weights, so they are evaluated in Python here and the
emitted graph contains only model nodes. The preset table values
(`Turbo` 12/0.5/1.75, `Default` 20/0.0/1.75, `Quality` 48/0.0/1.5) and the CFG
override (`cfg=3` from 70%) come from that template verbatim.

Node ids are descriptive strings (`load_unet`, `guider`) rather than the numeric
ones the ComfyUI frontend emits. The backend treats prompt keys as opaque, so
this is safe and makes the graph and its error messages readable.

## Non-obvious behaviours

**Two transformers are resident at once.** `DualModelGuider` runs the positive
pass on `ideogram4_fp8_scaled` and the negative pass on
`ideogram4_unconditional_fp8_scaled`, both 9.28 GB. With the 10.59 GB Qwen3-VL
encoder and the VAE that is 29.5 GB on disk, of which ~18.9 GB stays resident
through sampling once the encoder is offloaded. Anything with 24 GB or less has
to swap a transformer per step, which is why the floor is a 48 GB card.

`model_negative` is optional on `DualModelGuider` — dropping it falls back to
ordinary single-model CFG and halves transformer VRAM. That is a memory lever,
not a compute one (CFG still runs two passes), and it changes output, since the
unconditional transformer is separately trained.

**`mu`/`std` are not a shift value.** They parameterise the logit-normal
schedule in `Ideogram4Scheduler`; `mean = mu + 0.5 * log(w*h / 512²)`. Copying
a shift number from a Flux workflow will produce garbage.

**`max_containers` defaults to 1 on purpose.** ComfyUI's queue, `/history` and
`/view` are per-container state. A second replica would happily accept a
`/history/<id>` poll for a prompt it never ran and answer "not found". Raising
it is only safe for clients that submit and collect within one request — which
`/generate` does, but the raw proxy path does not.

**The models Volume is mounted at `/root/models`, not `ComfyUI/models`.** A
mount over the latter would hide the config YAMLs the repo ships there. An
`extra_model_paths.yaml` baked into the image adds the Volume as a search path
instead. Serving containers mount it read-only; only `download_models` writes.

**The proxy strips hop-by-hop headers and nothing else.** `content-length` and
`content-encoding` are dropped because the body is re-streamed. Any further
rewriting would break the ComfyUI frontend, which is sensitive about its own
routes. ComfyUI mirrors every route under `/api`, so both `/ws` and `/api/ws`
are registered as WebSocket proxies.

**Route order is load-bearing.** FastAPI matches in registration order and the
catch-all proxy is `/{path:path}`. Anything added after it is unreachable; new
typed endpoints must be registered before it in `create_app`.

**Credentials never reach the workflow JSON.** The custom node reads
`IDEOGRAM4_MODAL_URL` / `MODAL_KEY` / `MODAL_SECRET` from the environment or a
`.env` beside itself, so exported ComfyUI workflows stay shareable. Keep it that
way — do not add a token widget.

**Progress crosses the boundary via `client_id`, not a side channel.** ComfyUI
addresses progress events to the client id that submitted the prompt, so
`/generate` accepts one and forwards it verbatim. The node passes an id, then
subscribes to the proxied `/ws?clientId=<id>` and replays `progress` events into
a local `comfy.utils.ProgressBar`. No extra endpoint and no server-side state.

That mirror runs on a daemon thread and swallows every exception it raises: a
broken socket must never fail a render that succeeded. The corollary is that a
bug in it would be silent, which is why `tests/test_progress_mirror.py` drives
it against a real websocket server rather than trusting it.

Its teardown latency is on the critical path — `__exit__` runs after the image
is already in hand — so `POLL_INTERVAL_S` is the stop-check interval and the
test asserts setup plus teardown stays under a second.

**Plain-text prompts are effectively broken, not merely degraded.** Ideogram 4
conditions on a structured JSON caption schema. Free text is encoded and passed
through without error, and yields a coherent image bearing little relation to the
prompt — a silent failure that looks like a wiring bug. It is not: the graph here
matches ComfyUI's own `blueprints/Text to Image (Ideogram v4).json` node for
node. `assets/example_json_prompt.json` is the reference caption from that
template and is the fastest way to prove the pipeline before suspecting it.

**Config is deploy-time, not runtime.** `IDEOGRAM4_*` variables are read when
`modal deploy` executes `app.py`. Changing them requires a re-deploy.

## Testing strategy

`tests/test_workflow.py` covers `workflow.py` offline: preset application,
dimension snapping, link integrity, the dual-branch wiring. It deliberately does
**not** assert node input names against a hard-coded schema copy, because that
copy would rot.

`tests/test_server.py` drives the real ASGI app against a stub ComfyUI over an
in-memory transport, so the submit/poll/fetch sequence and the proxy's
transparency (request bodies forwarded, `Content-Encoding` preserved on raw
pass-through) are covered without a GPU. Add to the stub rather than mocking
`httpx` — the point is to exercise the wire format.

Schema agreement is checked against reality instead: `client.py validate` pulls
`/object_info` from the deployment and verifies every `class_type` and input
name in the built graph. Run it after bumping `COMFYUI_REF`.

## Known gaps

- Only text-to-image. Ideogram 4 also does editing and reference-image
  conditioning; those need `/upload/image` plumbing in the typed contract (the
  raw proxy already exposes the endpoint).
- No image-to-image, no LoRA loading.
- Cold start is dominated by moving ~30 GB from the Volume to VRAM. Modal's GPU
  memory snapshots would help and are not enabled here.
