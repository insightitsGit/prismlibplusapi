#!/usr/bin/env python3
"""
examples/enterprise_server.py
==============================

Runnable enterprise PrismAPI server with:
  - API key auth + rate limiting + audit log
  - GET /health, GET /metrics (Prometheus), GET /audit/recent
  - POST /chorus/search  (CHORUS binary frames)

Run (development):
    pip install "prismlib-plus[enterprise,cache]"
    python examples/enterprise_server.py

Production:
    export PRISM_API_KEYS="$(python -c 'from prism.api import generate_api_key; print(generate_api_key())')"
    export PRISM_API_REQUIRE_AUTH=true
    export PRISM_TENANT_ID=my-tenant
    uvicorn examples.enterprise_server:app --host 0.0.0.0 --port 9100

Or:
    python examples/enterprise_server.py --host 0.0.0.0 --port 9100
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running as script: python examples/enterprise_server.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prism.api import AuthConfig, PrismAPIProvider, generate_api_key
from prism.api.schema import SentenceTransformerEmbedder
from prism.enterprise import create_enterprise_app
from prism.lib.lang import PrismProjector, ProjectionConfig

logger = logging.getLogger(__name__)

TENANT_ID = os.getenv("PRISM_TENANT_ID", "enterprise-demo")

DOCS = [
    {"doc_id": "d1", "title": "Return policy", "body": "30-day returns on all items.", "price": 0},
    {"doc_id": "d2", "title": "Shipping", "body": "Free shipping on orders over $50.", "price": 50},
    {"doc_id": "d3", "title": "Warranty", "body": "12-month manufacturer warranty included.", "price": 0},
    {"doc_id": "d4", "title": "Support hours", "body": "Live chat Mon–Fri 9am–6pm EST.", "price": 0},
]


def _build_embedder():
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        from prism.cache.embedder import HashEmbedder
        logger.warning("sentence-transformers unavailable — using HashEmbedder")
        return HashEmbedder(output_dim=384)


def build_app(*, auth: AuthConfig | None = None, quiet: bool = False):
    """Factory used by uvicorn: uvicorn examples.enterprise_server:app"""
    projector = PrismProjector(ProjectionConfig(tenant_id=TENANT_ID, target_dim=64))
    embedder = _build_embedder()
    provider = PrismAPIProvider(
        projector=projector,
        embedder=embedder,
        semantic_fields=["title", "body"],
        id_field="doc_id",
        exact_fields=["price"],
    )

    @provider.expose
    def search(query: str, top_k: int = 10) -> list[dict]:
        q = query.lower()
        hits = [
            d for d in DOCS
            if q in d["title"].lower() or q in d["body"].lower()
        ]
        return hits[:top_k]

    if auth is None:
        env_auth = AuthConfig.from_env()
        if env_auth.enabled:
            auth = env_auth
        else:
            dev_key = generate_api_key()
            auth = AuthConfig(api_keys=(dev_key,), require_auth=True)
            if not quiet:
                logger.warning("No PRISM_API_KEYS set — generated dev key: %s", dev_key)
                print(f"\n  Dev API key (X-API-Key): {dev_key}\n", flush=True)

    return create_enterprise_app(
        provider=provider,
        handler=search,
        handler_name="search",
        auth=auth,
        title=f"PrismAPI Enterprise ({TENANT_ID})",
    )


# Uvicorn entrypoint: uvicorn examples.enterprise_server:app
app = build_app(quiet=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PrismAPI enterprise server")
    parser.add_argument("--host", default=os.getenv("PRISM_HTTP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PRISM_HTTP_PORT", "9100")))
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit('pip install "prismlib-plus[enterprise]" uvicorn') from exc

    print(f"PrismAPI enterprise server → http://{args.host}:{args.port}")
    print("  POST /chorus/search   (Content-Type: application/x-chorus-frame)")
    print("  GET  /health  /metrics  /audit/recent")
    if args.reload:
        uvicorn.run(
            "examples.enterprise_server:app",
            host=args.host,
            port=args.port,
            reload=True,
            factory=False,
        )
    else:
        uvicorn.run(build_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
