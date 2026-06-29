"""
prism.api — Vector-native API layer for AI agents
==================================================

PrismAPI lets an API serve content that is ALREADY embedded and projected
into PrismResonance space, delivered over CHORUS as raw float32 vectors.
The consuming agent retrieves results directly with no JSON parsing and no
re-embedding call.

Quick start::

    # Provider side (one line to adopt)
    from prism.api import PrismAPIProvider
    from prism.api.schema import SentenceTransformerEmbedder
    from prism.lib.lang import PrismProjector, ProjectionConfig

    projector = PrismProjector(ProjectionConfig(tenant_id="my-tenant"))
    embedder  = SentenceTransformerEmbedder()
    provider  = PrismAPIProvider(projector, embedder,
                                 semantic_fields=["title", "body"],
                                 id_field="doc_id")

    @provider.expose
    def search(query: str, top_k: int = 10) -> list[dict]:
        return my_db.search(query, top_k)   # unchanged

    # Consumer side
    from prism.api import PrismAPIClient

    client = PrismAPIClient(projector, embedder, loopback_provider=provider)
    response = client.query("how does inflation affect bonds?", top_k=5)
    # response.vectors  → np.ndarray (5, 64) — ready for PrismResonance
    # response.sidecars → list of exact metadata dicts — no re-embedding

See prism/api/README.md for full documentation and benchmark numbers.
"""

from prism.api.provider import ASGIAdapter, ExposedHandler, PrismAPIProvider
from prism.api.consumer import LangGraphTool, PrismAPIClient, RetryConfig
from prism.api.auth import AuthConfig, AuthError, generate_api_key
from prism.api.multi_provider import MultiProviderClient, MultiProviderResponse
from prism.api.schema import (
    APIRequest,
    APIResponse,
    Embedder,
    ExactSidecar,
    SemanticItem,
    SentenceTransformerEmbedder,
)
from prism.api.integrations.langgraph import (
    PrismRetrieverNode,
    MultiProviderRetrieverNode,
    create_retriever_node,
)

__all__ = [
    # Provider
    "PrismAPIProvider",
    "ExposedHandler",
    "ASGIAdapter",
    # Consumer
    "PrismAPIClient",
    "RetryConfig",
    "LangGraphTool",
    # Multi-provider
    "MultiProviderClient",
    "MultiProviderResponse",
    # LangGraph integration
    "PrismRetrieverNode",
    "MultiProviderRetrieverNode",
    "create_retriever_node",
    # Schema
    "Embedder",
    "SentenceTransformerEmbedder",
    "SemanticItem",
    "ExactSidecar",
    "APIRequest",
    "APIResponse",
    # Auth
    "AuthConfig",
    "AuthError",
    "generate_api_key",
]
