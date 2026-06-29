"""
examples/prismapi_quickstart.py
================================

PrismAPI end-to-end quickstart: provider patch + agent consumer.

Demonstrates:
    1. Decorating an existing handler with @provider.expose (one line).
    2. An LangGraph-style consumer querying over the CHORUS wire (loopback).
    3. Comparing the result with a plain HTTP-style baseline response.

Run:
    pip install sentence-transformers numpy
    python examples/prismapi_quickstart.py

No network required — the demo uses loopback mode (frames serialised
and deserialised in-process through the same CHORUSFrame wire path).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prism.lib.lang import PrismProjector, ProjectionConfig
from prism.api import (
    PrismAPIProvider,
    PrismAPIClient,
    LangGraphTool,
    SentenceTransformerEmbedder,
)

# ---------------------------------------------------------------------------
# Sample data — this would be your database in production
# ---------------------------------------------------------------------------

DOCS = [
    {"id": "doc_001", "title": "CHORUS Fabric overview",
     "body": "CHORUS Fabric is a tensor-native communication protocol for AI agents. "
             "It streams float32 vectors directly without JSON serialization, using a "
             "matrix-multiply cipher (V_enc = V_raw @ K) for built-in encryption.",
     "type": "technical", "published": True},
    {"id": "doc_002", "title": "Benchmark results",
     "body": "Transatlantic benchmark: US East to Frankfurt, 7,766 transmissions. "
             "4.45x less bandwidth than HTTP/REST. 179ms p50 latency — matches the "
             "physical speed-of-light minimum for the distance.",
     "type": "benchmark", "published": True},
    {"id": "doc_003", "title": "Orthogonal channel sharing",
     "body": "Two agents can share one gRPC channel using orthogonal weight matrices. "
             "W_A @ W_B ≈ 0 guarantees crosstalk below 0.000006%. Perfect signal "
             "recovery for both agents on a single transport.",
     "type": "technical", "published": True},
    {"id": "doc_004", "title": "Zero-knowledge relay",
     "body": "The relay node amplifies ciphertext without holding the decryption key. "
             "It produces a SHA-256 audit log of every frame it relays, enabling "
             "compliance and forensic reconstruction.",
     "type": "architecture", "published": True},
    {"id": "doc_005", "title": "PrismAPI vector layer",
     "body": "PrismAPI is a vector-native API layer. Providers pre-embed content and "
             "serve pre-projected float32 vectors over CHORUS. Consumers receive results "
             "without running a single embedding call on their side.",
     "type": "product", "published": True},
    {"id": "doc_006", "title": "LangGraph integration",
     "body": "PrismAPIClient integrates with LangGraph as a tool node. The tool "
             "embeds the query once, sends a CHORUS API_REQUEST, and returns "
             "pre-projected vectors ready for PrismResonance retrieval.",
     "type": "integration", "published": True},
    {"id": "doc_007", "title": "Johnson-Lindenstrauss projection",
     "body": "PrismProjector reduces embeddings from 384 dimensions to 64 using a "
             "Johnson-Lindenstrauss random projection matrix seeded by "
             "SHA-256(tenant_id). This provides tenant isolation and 6x memory reduction.",
     "type": "technical", "published": True},
    {"id": "doc_008", "title": "Patent pending",
     "body": "The Chorus Fabric protocol is patent pending. USPTO provisional "
             "application number 64/096,156 covers the tensor-native communication "
             "method and matrix-multiply cipher architecture.",
     "type": "legal", "published": True},
]


# ---------------------------------------------------------------------------
# Step 1: Existing handler (no changes needed)
# ---------------------------------------------------------------------------

def existing_search_handler(query: str = "", top_k: int = 5) -> list[dict]:
    """Your existing search handler — unchanged."""
    # In production: full-text search, database query, etc.
    # Here: simple substring match for demonstration.
    query_lower = query.lower()
    results = [
        doc for doc in DOCS
        if any(
            query_lower in doc.get(field, "").lower()
            for field in ["title", "body"]
        )
        and doc.get("published", False)
    ]
    return results[:top_k] if results else DOCS[:top_k]


# ---------------------------------------------------------------------------
# Step 2: Patch the handler with @provider.expose (one decorator line)
# ---------------------------------------------------------------------------

def setup_provider(embedder: SentenceTransformerEmbedder, target_dim: int = 64):
    projector = PrismProjector(ProjectionConfig(
        tenant_id="quickstart-tenant",
        target_dim=target_dim,
    ))
    provider = PrismAPIProvider(
        projector=projector,
        embedder=embedder,
        semantic_fields=["title", "body"],   # vectorized
        id_field="id",
        exact_fields=["type", "published"],   # never vectorized — ride as sidecar
        provider_id="quickstart-provider",
    )

    # One line patch — the handler itself is UNCHANGED
    @provider.expose
    def search(query: str = "", top_k: int = 5) -> list[dict]:
        return existing_search_handler(query=query, top_k=top_k)

    return provider, projector


# ---------------------------------------------------------------------------
# Step 3: Consumer (LangGraph-style tool node)
# ---------------------------------------------------------------------------

def run_demo(query: str = "How does CHORUS handle encryption?") -> None:
    print("\n" + "=" * 60)
    print("PrismAPI Quickstart")
    print("=" * 60)

    # Load embedder once (shared between provider and consumer)
    print("\nLoading embedder (all-MiniLM-L6-v2)...")
    try:
        embedder = SentenceTransformerEmbedder()
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("       pip install sentence-transformers")
        sys.exit(1)

    print(f"  Embed dim: {embedder.embed_dim}")

    # Provider setup
    provider, projector = setup_provider(embedder, target_dim=64)

    # --- Baseline: plain handler call (returns text, no vectors) ----------
    print(f"\n[BASELINE] Query: '{query}'")
    t0 = time.perf_counter()
    baseline_results = existing_search_handler(query=query, top_k=3)
    t_baseline = (time.perf_counter() - t0) * 1000.0
    print(f"  Results ({len(baseline_results)} docs, {t_baseline:.1f} ms):")
    for r in baseline_results:
        print(f"    [{r['id']}] {r['title']} (type={r['type']})")
    print("  → Consumer must now embed these texts to use them as vectors.")

    # --- PrismAPI: CHORUS loopback -------------------------------------------
    print(f"\n[PRISMAPI] Query: '{query}'")
    client = PrismAPIClient(
        projector=projector,
        embedder=embedder,
        loopback_provider=provider,
        source_field="body",
    )

    t0 = time.perf_counter()
    response = client.query(query, top_k=3)
    t_prismapi = (time.perf_counter() - t0) * 1000.0

    print(f"  Results ({len(response.results)} docs, {t_prismapi:.1f} ms):")
    for sem, side in response.results:
        print(f"    [{sem.doc_id}] {side.fields.get('type', '?')} · "
              f"vector dim={sem.vector.shape[0]} · "
              f"preview: '{sem.text_preview[:60]}...'")
    print(f"  → Vectors arrive pre-projected (64-d float32). No re-embedding needed.")
    print(f"  → Embedding calls saved: {response.embedding_calls_saved} "
          f"(would have been {len(response.results)} on the baseline path)")

    if len(response.results) > 0:
        print(f"\n  Stacked vector matrix shape: {response.vectors.shape}")
        print(f"  Sidecar fields: {list(response.sidecars[0].keys())}")

    # --- LangGraphTool usage -------------------------------------------------
    print(f"\n[LANGGRAPH TOOL] Same query via LangGraphTool.invoke()")
    tool = LangGraphTool(
        name="semantic_search",
        description="Search CHORUS Fabric documentation.",
        client=client,
        top_k=3,
    )

    t0 = time.perf_counter()
    tool_result = tool.invoke({"query": query})
    t_tool = (time.perf_counter() - t0) * 1000.0

    print(f"  tool_result['top_k']:               {tool_result['top_k']}")
    print(f"  tool_result['vectors'].shape:        {tool_result['vectors'].shape}")
    print(f"  tool_result['sidecars'][0]:          {tool_result['sidecars'][0]}")
    print(f"  tool_result['embedding_calls_saved']: {tool_result['embedding_calls_saved']}")
    print(f"  Latency: {t_tool:.1f} ms")

    # --- Wire frame size comparison ------------------------------------------
    import json
    baseline_json = json.dumps([
        {"id": r["id"], "title": r["title"], "body": r["body"]}
        for r in baseline_results
    ])
    chorus_frame = provider.as_chorus_frame(baseline_results)
    chorus_bytes = len(chorus_frame.to_bytes())
    json_bytes = len(baseline_json.encode())

    print(f"\n[WIRE SIZE COMPARISON]")
    print(f"  JSON response size:        {json_bytes:,} bytes")
    print(f"  CHORUSFrame size:          {chorus_bytes:,} bytes")
    print(f"  Reduction:                 "
          f"{100.0 * (1 - chorus_bytes / json_bytes):.1f}%")

    print("\n" + "=" * 60)
    print("Done. See benchmark/api/run_prismapi_benchmark.py for full numbers.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How does CHORUS handle encryption?"
    run_demo(query)
