"""
prism.bridge — Vector store adapters and RAG patch layer.

The bridge module provides connectors to external vector databases
(pgvector, Chroma, Qdrant) and the PrismRAGPatch taxonomy classification
pipeline that upgrades raw embeddings with semantic category anchoring
before insertion.

The old direct-DB connectors (PostgreSQLConnector, MySQLConnector, WAL
adapters) have moved to prism.wrapper — they now live on the DB node
inside the Server Wrapper daemon, not in the application process.
"""

from prism.bridge.vector import (
    VectorStoreAdapter,
    PrismRAGPatch,
    PgVectorAdapter,
    ChromaAdapter,
    QdrantAdapter,
    TaxonomyCategory,
    PatchedVector,
)

__all__ = [
    "VectorStoreAdapter",
    "PrismRAGPatch",
    "PgVectorAdapter",
    "ChromaAdapter",
    "QdrantAdapter",
    "TaxonomyCategory",
    "PatchedVector",
]
