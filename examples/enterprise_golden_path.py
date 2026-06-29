"""
examples/enterprise_golden_path.py
===================================

End-to-end enterprise stack (loopback / in-process):

  PrismAPI provider  →  PrismAPIClient retrieve
  PrismCache         →  semantic LLM dedup
  LocalIndex         →  WAL upsert/delete (driver replica)
  AuthConfig         →  API key gate for ASGI endpoints
  MetricsRegistry    →  Prometheus-ready counters

Run:
    python examples/enterprise_golden_path.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prism.api import PrismAPIProvider, PrismAPIClient, AuthConfig, generate_api_key
from prism.api.schema import SentenceTransformerEmbedder
from prism.cache import PrismCache
from prism.ffi.bindings import LocalIndex
from prism.lib.lang import PrismProjector, ProjectionConfig
from prism.observability import record_cache_hit, record_driver_index, get_registry

DOCS = [
    {"doc_id": "d1", "title": "Returns", "body": "30-day return policy for all items."},
    {"doc_id": "d2", "title": "Shipping", "body": "Free shipping over $50."},
]


def main() -> None:
    tenant = "enterprise-demo"
    projector = PrismProjector(ProjectionConfig(tenant_id=tenant, target_dim=64))
    try:
        embedder = SentenceTransformerEmbedder()
    except Exception:
        from prism.cache.embedder import HashEmbedder
        embedder = HashEmbedder(output_dim=64)

    provider = PrismAPIProvider(
        projector=projector,
        embedder=embedder,
        semantic_fields=["title", "body"],
        id_field="doc_id",
    )

    @provider.expose
    def search(query: str, top_k: int = 5) -> list[dict]:
        q = query.lower()
        return [d for d in DOCS if q in d["title"].lower() or q in d["body"].lower()][:top_k]

    api_key = generate_api_key()
    auth = AuthConfig(api_keys=(api_key,), require_auth=True)
    print(f"API key (dev): {api_key[:8]}...")

    client = PrismAPIClient(projector, embedder, loopback_provider=provider)
    response = client.query("return policy", top_k=3)
    print(f"PrismAPI: {len(response.items)} vector results")

    cache = PrismCache.build(tenant_id=tenant, llm_model="demo")
    answer = cache.get_or_call(
        query="What is the return policy?",
        call_fn=lambda: "30-day returns on all items.",
        metadata={"doc_id": "d1"},
    )
    print(f"PrismCache answer: {answer}")
    record_cache_hit(latency_ms=0.5)

    idx = LocalIndex(tenant_id=tenant)
    for doc in DOCS:
        text = f"{doc['title']}: {doc['body']}"
        raw = embedder.embed(text)
        env = projector.project(raw)
        idx.apply_event(
            event_id=doc["doc_id"],
            row_id=doc["doc_id"],
            op="INSERT",
            text_repr=text,
            vector=env.vector.tolist(),
        )
    qvec = projector.project(embedder.embed("shipping cost")).vector
    hits, ms = idx.query(qvec, top_k=2, threshold=0.3)
    print(f"LocalIndex: {len(hits)} hits in {ms:.3f}ms")
    record_driver_index(size=idx.size, rows_received=idx.rows_received, rows_deleted=idx.rows_deleted)

    print("\nPrometheus sample:")
    print(get_registry().to_prometheus()[:400], "...")
    print("\nGolden path OK.")


if __name__ == "__main__":
    main()
