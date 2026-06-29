"""
Example: PrismCache with OpenAI GPT-4o
=======================================

Install:
    pip install prismlib openai sentence-transformers

Run:
    OPENAI_API_KEY=sk-... python examples/openai_basic.py
"""

import os
import time
from openai import OpenAI
from prism.cache import PrismCache

# ── 1. Build the cache ────────────────────────────────────────────────────────
cache = PrismCache.build(
    tenant_id="my-company",
    llm_model="gpt-4o",
    similarity_threshold=0.92,
    ttl_seconds=3600,
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── 2. Wrap your LLM call ─────────────────────────────────────────────────────
def ask(question: str) -> str:
    return cache.get_or_call(
        query=question,
        call_fn=lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
        ).choices[0].message.content,
        tokens_in_response=300,
    )


# ── 3. Use it ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What is your return policy?",
        "How do I return an item?",          # semantically similar → cache hit
        "Can I get a refund on my order?",   # semantically similar → cache hit
        "What are your business hours?",     # different topic → cache miss
        "When are you open?",                # similar to above → cache hit
    ]

    for q in questions:
        t0 = time.perf_counter()
        answer = ask(q)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[{elapsed:6.1f}ms] Q: {q[:50]}")
        print(f"           A: {str(answer)[:80]}...")
        print()

    # ── 4. See the savings ────────────────────────────────────────────────────
    cache.print_metrics()
