"""
Example: Multi-tenant PrismCache
=================================

Each tenant gets a mathematically isolated cache space.
Tenant A's cached answers are invisible to Tenant B — not by
database filtering, but by the JL projection math.

This is the key SaaS use case: one deployment, many customers,
zero cross-tenant data leakage.

Install:
    pip install prismlib sentence-transformers

Run:
    python examples/multi_tenant.py
"""

import time
from prism.cache import PrismCache, HashEmbedder

# Simulate different customers of your SaaS product
TENANTS = ["acme-corp", "globex-inc", "initech-llc"]


def build_tenant_cache(tenant_id: str) -> PrismCache:
    """Each tenant gets its own PrismCache with an isolated projection space."""
    return PrismCache.build(
        tenant_id=tenant_id,
        llm_model="gpt-4o-mini",
        similarity_threshold=0.90,
        # Using HashEmbedder for this demo (no API key needed)
        # In production: SentenceTransformerEmbedder() or OpenAIEmbedder(...)
        embedder=HashEmbedder(output_dim=384),
    )


def simulate_llm(prompt: str, tenant: str) -> str:
    """Fake LLM that returns tenant-specific answers."""
    time.sleep(0.05)  # simulate 50ms LLM latency
    return f"[{tenant}] Answer to: {prompt[:40]}"


if __name__ == "__main__":
    caches = {t: build_tenant_cache(t) for t in TENANTS}

    # Each tenant caches their own queries
    for tenant_id, cache in caches.items():
        print(f"\n── Tenant: {tenant_id} ──")

        questions = [
            "What is our Q3 revenue forecast?",
            "Show me Q3 revenue projections.",    # similar → should hit for same tenant
            "What is the headcount plan?",
        ]

        for q in questions:
            t0 = time.perf_counter()
            answer = cache.get_or_call(
                query=q,
                call_fn=lambda q=q, t=tenant_id: simulate_llm(q, t),
            )
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [{elapsed:5.1f}ms] {q[:45]:<45} → {answer[:30]}")

    # Show that querying Tenant A's question from Tenant B's cache is a miss
    print("\n── Cross-tenant isolation test ──")
    acme_cache = caches["acme-corp"]
    globex_cache = caches["globex-inc"]

    # Prime acme's cache
    acme_cache.get_or_call(
        query="What is our annual recurring revenue?",
        call_fn=lambda: simulate_llm("ARR question", "acme-corp"),
    )

    # Globex asks the same question — should be a MISS (isolated space)
    t0 = time.perf_counter()
    result = globex_cache.get_or_call(
        query="What is our annual recurring revenue?",
        call_fn=lambda: simulate_llm("ARR question", "globex-inc"),
    )
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  Acme primed cache with 'annual recurring revenue'")
    print(f"  Globex queried same question → [{elapsed:.1f}ms] (expect ~50ms miss)")
    print(f"  Result: {result}")
    print()

    # Metrics per tenant
    print("── Per-tenant metrics ──")
    for tenant_id, cache in caches.items():
        m = cache.get_metrics()
        print(f"  {tenant_id}: {m.hit_rate_pct:.0f}% hit rate, "
              f"{m.total_queries} queries, "
              f"${m.total_cost_saved_usd:.4f} saved")
