#!/usr/bin/env python3
"""Command-line client for the deployed FLUX.2 klein 9B endpoint.

export FLUX2KLEIN_MODAL_URL=https://...modal.run
export MODAL_KEY=wk-...  MODAL_SECRET=ws-...

./client.py generate "a neon ramen shop in the rain" --aspect-ratio 16:9
./client.py generate "a portrait" --variant distilled      # 4 steps, cfg 1
./client.py variants
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

ENV_URL = "FLUX2KLEIN_MODAL_URL"


def main() -> None:
    parser, subparsers = cli.build_parser(__doc__, ENV_URL)

    gen = subparsers.add_parser("generate", help="render an image")
    gen.add_argument("prompt")
    gen.add_argument("--negative", default="", help="base variant only")
    gen.add_argument(
        "--variant", default=workflow.DEFAULT_VARIANT, choices=sorted(workflow.VARIANTS)
    )
    gen.add_argument("--lora", choices=sorted(workflow.LORAS), help="layer an adapter on top")
    # No default: omitting it lets the server apply the adapter's own
    # recommended strength rather than a one-size-fits-all 1.0.
    gen.add_argument(
        "--lora-strength",
        type=float,
        help="defaults to the adapter's recommended strength; see `variants`",
    )
    gen.add_argument("--steps", type=int, help="overrides the variant default")
    gen.add_argument("--cfg", type=float, help="overrides the variant default")
    cli.add_geometry_arguments(gen, workflow.ASPECT_RATIOS)

    subparsers.add_parser("variants", help="list variants and their sampler defaults")

    args = parser.parse_args()
    url = cli.endpoint(args.url, ENV_URL)

    if args.command == "health":
        return cli.cmd_health(url)

    if args.command == "validate":
        graph = workflow.build_workflow(workflow.resolve_params("validation probe"))
        return cli.cmd_validate(url, graph)

    if args.command == "variants":
        print(json.dumps(cli.get_json(url, "/variants"), indent=2))
        return None

    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative,
        "variant": args.variant,
        **cli.geometry_payload(args),
    }
    if args.lora:
        payload["lora"] = args.lora
        if args.lora_strength is not None:
            payload["lora_strength"] = args.lora_strength
    # Left unset, the server applies the variant's own steps/cfg.
    if args.steps is not None:
        payload["steps"] = args.steps
    if args.cfg is not None:
        payload["cfg"] = args.cfg

    result = cli.generate(url, payload, args.out, args.timeout)
    params = result["params"]
    print(
        f"seed={params['seed']} variant={params['variant']} steps={params['steps']} "
        f"cfg={params['cfg']} {params['width']}x{params['height']} in {result['duration_s']}s"
    )
    return None


if __name__ == "__main__":
    main()
