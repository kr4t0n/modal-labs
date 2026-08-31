#!/usr/bin/env python3
"""Command-line client for the deployed Ideogram 4 endpoint.

export IDEOGRAM4_MODAL_URL=https://...modal.run
export MODAL_KEY=wk-...  MODAL_SECRET=ws-...

./client.py generate "a neon ramen shop in the rain" --aspect-ratio 16:9
./client.py generate --json-prompt caption.json --preset Quality
./client.py template "a neon ramen shop in the rain" > caption_prompt.txt
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

ENV_URL = "IDEOGRAM4_MODAL_URL"


def main() -> None:
    parser, subparsers = cli.build_parser(__doc__, ENV_URL)

    gen = subparsers.add_parser("generate", help="render an image")
    gen.add_argument("prompt", nargs="?", default="")
    gen.add_argument("--json-prompt", help="path to a structured Ideogram 4 caption")
    gen.add_argument(
        "--preset", default=workflow.DEFAULT_PRESET, choices=sorted(workflow.SAMPLING_PRESETS)
    )
    gen.add_argument("--cfg", type=float, default=workflow.DEFAULT_CFG)
    cli.add_geometry_arguments(gen, workflow.ASPECT_RATIOS)

    tmpl = subparsers.add_parser("template", help="print the magic-prompt template for an LLM")
    tmpl.add_argument("prompt")
    tmpl.add_argument("--width", type=int, default=1024)
    tmpl.add_argument("--height", type=int, default=1024)

    args = parser.parse_args()
    url = cli.endpoint(args.url, ENV_URL)

    if args.command == "health":
        return cli.cmd_health(url)

    if args.command == "validate":
        graph = workflow.build_workflow(workflow.resolve_params("validation probe"))
        return cli.cmd_validate(url, graph)

    if args.command == "template":
        sys.stdout.write(
            cli.get_text(
                url,
                "/caption-template",
                {"prompt": args.prompt, "width": args.width, "height": args.height},
            )
        )
        return None

    if not args.prompt and not args.json_prompt:
        sys.exit("provide a prompt or --json-prompt")

    payload = {"preset": args.preset, "cfg": args.cfg, **cli.geometry_payload(args)}
    if args.json_prompt:
        payload["json_prompt"] = json.loads(Path(args.json_prompt).read_text(encoding="utf-8"))
    else:
        payload["prompt"] = args.prompt

    result = cli.generate(url, payload, args.out, args.timeout)
    params = result["params"]
    print(
        f"seed={params['seed']} steps={params['steps']} "
        f"{params['width']}x{params['height']} in {result['duration_s']}s"
    )
    return None


if __name__ == "__main__":
    main()
