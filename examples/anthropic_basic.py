"""
Example: PrismCache with Anthropic Claude
==========================================

Install:
    pip install prismlib anthropic voyageai sentence-transformers

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/anthropic_basic.py
"""

import asyncio
import os
import anthropic
from prism.cache import PrismCache, AnthropicEmbedder

# ── 1. Build the cache with Voyage AI embeddings (Anthropic's embedder) ───────
cache = PrismCache.build(
    tenant_id="my-company",
    llm_model="claude-sonnet-4-6",
    similarity_threshold=0.92,
    embedder=AnthropicEmbedder(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="voyage-3",
    ),
)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── 2. Async usage ────────────────────────────────────────────────────────────
async def ask(question: str) -> str:
    async def call_claude() -> str:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": question}],
        )
        return message.content[0].text

    return await cache.aget_or_call(
        query=question,
        call_fn=call_claude,
        tokens_in_response=400,
    )


async def main() -> None:
    async with cache:
        questions = [
            "Explain quantum entanglement simply.",
            "What is quantum entanglement in simple terms?",   # hit
            "Give me a simple explanation of quantum entanglement.",  # hit
            "How does photosynthesis work?",                   # miss
        ]

        for q in questions:
            answer = await ask(q)
            print(f"Q: {q}")
            print(f"A: {str(answer)[:120]}...")
            print()

        cache.print_metrics()


if __name__ == "__main__":
    asyncio.run(main())
