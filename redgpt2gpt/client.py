#!/usr/bin/env python3
"""Command-line client for the deployed RedGPT2 endpoint.

export REDGPT2GPT_MODAL_URL=https://...modal.run
export MODAL_KEY=wk-...  MODAL_SECRET=ws-...

./client.py generate "a portrait by a window" --aspect-ratio 3:4
./client.py generate "..." --cfg 3 --steps 20 --negative "blurry"
./client.py defaults
./client.py validate
./client.py health
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import workflow
from comfyui_modal import cli

ENV_URL = "REDGPT2GPT_MODAL_URL"


def main() -> None:
    parser, subparsers = cli.build_parser(__doc__, ENV_URL)

    gen = subparsers.add_parser("generate", help="render an image")
    gen.add_argument("prompt")
    gen.add_argument("--negative", default=None, help="only meaningful alongside a raised --cfg")
    gen.add_argument("--steps", type=int, default=workflow.DEFAULT_STEPS)
    gen.add_argument("--cfg", type=float, default=workflow.DEFAULT_CFG)
    gen.add_argument("--sampler", default=workflow.DEFAULT_SAMPLER)
    gen.add_argument("--scheduler", default=workflow.DEFAULT_SCHEDULER)
    cli.add_geometry_arguments(gen, workflow.ASPECT_RATIOS)

    subparsers.add_parser("defaults", help="show the sampler conventions this service applies")

    args = parser.parse_args()
    url = cli.endpoint(args.url, ENV_URL)

    if args.command == "health":
        return cli.cmd_health(url)

    if args.command == "validate":
        graph = workflow.build_workflow(workflow.resolve_params("validation probe"))
        return cli.cmd_validate(url, graph)

    if args.command == "defaults":
        print(json.dumps(cli.get_json(url, "/defaults"), indent=2))
        return None

    payload = {
        "prompt": args.prompt,
        "steps": args.steps,
        "cfg": args.cfg,
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        **cli.geometry_payload(args),
    }
    # Left unset, the server applies its own default negative.
    if args.negative is not None:
        payload["negative_prompt"] = args.negative

    result = cli.generate(url, payload, args.out, args.timeout)
    params = result["params"]
    print(
        f"seed={params['seed']} steps={params['steps']} cfg={params['cfg']} "
        f"{params['sampler_name']}/{params['scheduler']} "
        f"{params['width']}x{params['height']} in {result['duration_s']}s"
    )
    return None


if __name__ == "__main__":
    main()
