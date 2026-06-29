#!/usr/bin/env python3
"""
examples/enterprise_client.py
==============================

HTTP client for a running PrismAPI enterprise server.

Prerequisites:
    1. Start the server:  python examples/enterprise_server.py
    2. Copy the dev API key printed at startup (or set PRISM_API_KEY).

Usage:
    python examples/enterprise_client.py --query "return policy"
    PRISM_API_KEY=... python examples/enterprise_client.py --health-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prism.api import PrismAPIClient
from prism.api.schema import SentenceTransformerEmbedder
from prism.lib.lang import PrismProjector, ProjectionConfig


def _embedder():
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        from prism.cache.embedder import HashEmbedder
        return HashEmbedder(output_dim=384)


def main() -> None:
    parser = argparse.ArgumentParser(description="PrismAPI enterprise HTTP client")
    parser.add_argument("--host", default=os.getenv("PRISM_HTTP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PRISM_HTTP_PORT", "9100")))
    parser.add_argument("--api-key", default=os.getenv("PRISM_API_KEY", ""))
    parser.add_argument("--query", default="return policy", help="Search query text")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--health-only", action="store_true")
    args = parser.parse_args()

    tenant = os.getenv("PRISM_TENANT_ID", "enterprise-demo")
    projector = PrismProjector(ProjectionConfig(tenant_id=tenant, target_dim=64))
    embedder = _embedder()

    client = PrismAPIClient(
        projector=projector,
        embedder=embedder,
        host=args.host,
        port=args.port,
        api_key=args.api_key or None,
    )

    try:
        if not client.health_check():
            raise SystemExit(
                f"Server not reachable at http://{args.host}:{args.port}/health"
            )
        print(f"Health OK — http://{args.host}:{args.port}")

        if args.health_only:
            return

        if not args.api_key:
            print(
                "Warning: no --api-key / PRISM_API_KEY — server may reject the request.",
                file=sys.stderr,
            )

        response = client.query(args.query, top_k=args.top_k)
        print(f"\nQuery: {args.query!r}")
        print(f"Results: {len(response.items)} ({response.latency_ms:.1f} ms)")
        for item, sidecar in response.results:
            print(f"  - {item.doc_id}: {item.text_preview[:80]!r}")
            if sidecar.fields:
                print(f"      sidecar: {sidecar.fields}")
        print("\nClient OK.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
