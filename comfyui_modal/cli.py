"""Shared plumbing for each service's command-line client.

A service's `client.py` supplies its own `generate` arguments and payload; the
transport, the health check, the schema validation and the argument scaffolding
live here.
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


def endpoint(override: str | None, env_var: str) -> str:
    url = (override or os.environ.get(env_var, "")).strip().rstrip("/")
    if not url:
        sys.exit(f"set {env_var} or pass --url")
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


def get_json(url: str, path: str, timeout: float = 120.0) -> Any:
    return check(requests.get(f"{url}{path}", headers=headers(), timeout=timeout)).json()


def get_text(url: str, path: str, params: dict[str, Any], timeout: float = 60.0) -> str:
    return check(
        requests.get(f"{url}{path}", params=params, headers=headers(), timeout=timeout)
    ).text


def generate(url: str, payload: dict[str, Any], out_dir: str, timeout: float) -> dict[str, Any]:
    """POST /generate, write the PNGs, and return the parsed response."""
    print(f"rendering on {url} ...", flush=True)
    result = check(
        requests.post(f"{url}/generate", json=payload, headers=headers(), timeout=timeout + 60)
    ).json()

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    params = result["params"]
    for index, record in enumerate(result["images"]):
        path = destination / f"{params.get('seed', 'image')}_{index}.png"
        path.write_bytes(base64.b64decode(record["b64"]))
        print(f"wrote {path}")
    return result


def cmd_health(url: str) -> None:
    print(json.dumps(get_json(url, "/health"), indent=2))


def cmd_validate(url: str, graph: dict[str, Any]) -> None:
    """Check a locally built graph against the remote ComfyUI's node schemas.

    Catches the failure mode that matters after a ComfyUI upgrade: a node whose
    input was renamed still builds fine locally and only fails at queue time.
    """
    schemas = get_json(url, "/object_info")

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


def add_geometry_arguments(
    parser: argparse.ArgumentParser,
    aspect_ratios,
    *,
    default_side: int = 1024,
    default_megapixels: float = 1.0,
) -> None:
    """The width/height/ratio/seed flags every client shares.

    `geometry_payload` always sends width and height, so a service whose model
    wants a different native resolution has to move these defaults rather than
    rely on the server's — otherwise the CLI overrides it on every call.
    """
    parser.add_argument("--aspect-ratio", choices=sorted(aspect_ratios))
    parser.add_argument("--megapixels", type=float, default=default_megapixels)
    parser.add_argument("--width", type=int, default=default_side)
    parser.add_argument("--height", type=int, default=default_side)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", default="outputs")


def geometry_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "width": args.width,
        "height": args.height,
        "batch_size": args.batch_size,
        "timeout_s": args.timeout,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.aspect_ratio:
        payload["aspect_ratio"] = args.aspect_ratio
        payload["megapixels"] = args.megapixels
    return payload


def build_parser(description: str, env_var: str):
    """A parser carrying --url plus the health/validate subcommands."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", help=f"defaults to ${env_var}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health", help="ping the deployment")
    subparsers.add_parser("validate", help="check the graph against the deployed node schemas")
    return parser, subparsers
