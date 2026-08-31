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

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).parent))

import workflow


def endpoint(override: str | None) -> str:
    url = (override or os.environ.get("FLUX2KLEIN_MODAL_URL", "")).strip().rstrip("/")
    if not url:
        sys.exit("set FLUX2KLEIN_MODAL_URL or pass --url")
    return url


def headers() -> dict[str, str]:
    key, secret = os.environ.get("MODAL_KEY"), os.environ.get("MODAL_SECRET")
    if key and secret:
        return {"Modal-Key": key, "Modal-Secret": secret}
    return {}


def check(response: requests.Response) -> requests.Response:
    if response.status_code >= 400:
        sys.exit(f"HTTP {response.status_code}: {response.text[:4000]}")
    return response


def cmd_generate(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative,
        "variant": args.variant,
        "width": args.width,
        "height": args.height,
        "batch_size": args.batch_size,
        "timeout_s": args.timeout,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    # Left unset, the server applies the variant's own steps/cfg.
    if args.steps is not None:
        payload["steps"] = args.steps
    if args.cfg is not None:
        payload["cfg"] = args.cfg
    if args.aspect_ratio:
        payload["aspect_ratio"] = args.aspect_ratio
        payload["megapixels"] = args.megapixels

    print(f"rendering on {args.url or os.environ.get('FLUX2KLEIN_MODAL_URL')} ...", flush=True)
    result = check(
        requests.post(
            f"{endpoint(args.url)}/generate",
            json=payload,
            headers=headers(),
            timeout=args.timeout + 60,
        )
    ).json()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = result["params"]
    for index, record in enumerate(result["images"]):
        path = out_dir / f"ideogram4_{params['seed']}_{index}.png"
        path.write_bytes(base64.b64decode(record["b64"]))
        print(f"wrote {path}")
    print(
        f"seed={params['seed']} variant={params['variant']} steps={params['steps']} "
        f"cfg={params['cfg']} {params['width']}x{params['height']} in {result['duration_s']}s"
    )


def cmd_variants(args: argparse.Namespace) -> None:
    response = check(requests.get(f"{endpoint(args.url)}/variants", headers=headers(), timeout=60))
    print(json.dumps(response.json(), indent=2))


def cmd_health(args: argparse.Namespace) -> None:
    response = check(requests.get(f"{endpoint(args.url)}/health", headers=headers(), timeout=120))
    print(json.dumps(response.json(), indent=2))


def cmd_validate(args: argparse.Namespace) -> None:
    """Check the local graph against the remote ComfyUI's node schemas.

    Catches the failure mode that matters after a ComfyUI upgrade: a node whose
    input was renamed still builds fine locally and only fails at queue time.
    """
    schemas = check(
        requests.get(f"{endpoint(args.url)}/object_info", headers=headers(), timeout=120)
    ).json()
    graph = workflow.build_workflow(workflow.resolve_params("validation probe"))

    problems = []
    for node_id, node in graph.items():
        schema = schemas.get(node["class_type"])
        if schema is None:
            problems.append(f"{node_id}: unknown class_type {node['class_type']!r}")
            continue
        accepted = set(schema.get("input", {}).get("required", {})) | set(
            schema.get("input", {}).get("optional", {})
        )
        for name in node["inputs"]:
            if name not in accepted:
                problems.append(
                    f"{node_id} ({node['class_type']}): input {name!r} not in {sorted(accepted)}"
                )

    for line in problems:
        print(f"FAIL {line}")
    if problems:
        sys.exit(f"{len(problems)} mismatch(es) against the deployed ComfyUI")
    print(f"OK — all {len(graph)} nodes match the deployed ComfyUI schemas")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", help="defaults to $FLUX2KLEIN_MODAL_URL")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="render an image")
    gen.add_argument("prompt")
    gen.add_argument("--negative", default="", help="base variant only")
    gen.add_argument(
        "--variant", default=workflow.DEFAULT_VARIANT, choices=sorted(workflow.VARIANTS)
    )
    gen.add_argument("--aspect-ratio", choices=sorted(workflow.ASPECT_RATIOS))
    gen.add_argument("--megapixels", type=float, default=1.0)
    gen.add_argument("--width", type=int, default=1024)
    gen.add_argument("--height", type=int, default=1024)
    gen.add_argument("--seed", type=int)
    gen.add_argument("--batch-size", type=int, default=1)
    gen.add_argument("--steps", type=int, help="overrides the variant default")
    gen.add_argument("--cfg", type=float, help="overrides the variant default")
    gen.add_argument("--timeout", type=float, default=900.0)
    gen.add_argument("--out", default="outputs")
    gen.set_defaults(func=cmd_generate)

    sub.add_parser("variants", help="list variants and their sampler defaults").set_defaults(
        func=cmd_variants
    )
    sub.add_parser("health", help="ping the deployment").set_defaults(func=cmd_health)
    sub.add_parser(
        "validate", help="check the graph against the deployed node schemas"
    ).set_defaults(func=cmd_validate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
