"""
PrismCache — In-Process Semantic LLM Cache
==========================================

Drop-in semantic cache for any LLM. Serves semantically similar queries
from local RAM in microseconds instead of calling the LLM again.

Quick start:
    from prism.cache import PrismCache

    cache = PrismCache.build(tenant_id="my-app", llm_model="gpt-4o")

    response = cache.get_or_call(
        query="What is your return policy?",
        call_fn=lambda: my_llm_client.chat("What is your return policy?"),
    )

    cache.print_metrics()
"""

from prism.cache.cache import PrismCache, PrismCacheConfig, CacheError
from prism.cache.embedder import (
    Embedder,
    SentenceTransformerEmbedder,
    OpenAIEmbedder,
    AnthropicEmbedder,
    OllamaEmbedder,
    HashEmbedder,
)
from prism.cache.store import CacheStore, InMemoryStore, SQLiteStore, CacheEntry
from prism.cache.metrics import CacheMetrics, CostModel, KNOWN_MODELS

__all__ = [
    # Main class
    "PrismCache",
    "PrismCacheConfig",
    "CacheError",
    # Embedders
    "Embedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "AnthropicEmbedder",
    "OllamaEmbedder",
    "HashEmbedder",
    # Stores
    "CacheStore",
    "InMemoryStore",
    "SQLiteStore",
    "CacheEntry",
    # Metrics
    "CacheMetrics",
    "CostModel",
    "KNOWN_MODELS",
]
