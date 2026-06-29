"""
Shared fixtures for prism.api tests.

All fixtures use a deterministic mock embedder — no sentence-transformers,
no network, no HuggingFace API calls.  Tests run offline in <1 second.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism.lib.lang import PrismProjector, ProjectionConfig


EMBED_DIM = 32     # small fake embedding dim
TARGET_DIM = 16    # small projected dim
TENANT_ID  = "test-tenant"


class MockEmbedder:
    """
    Deterministic fake embedder.
    Maps each text to a repeatable unit vector via a seeded hash.
    No ML model, no network.
    """

    embed_dim = EMBED_DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(self.embed_dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.stack(vecs, axis=0)


@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def projector() -> PrismProjector:
    return PrismProjector(ProjectionConfig(
        tenant_id=TENANT_ID,
        target_dim=TARGET_DIM,
    ))


SAMPLE_DOCS = [
    {"doc_id": "d1", "title": "Inflation basics", "body": "Inflation erodes purchasing power.", "category": "economics"},
    {"doc_id": "d2", "title": "Bond markets",     "body": "Bonds pay fixed coupons.", "category": "finance"},
    {"doc_id": "d3", "title": "Equity returns",   "body": "Stocks outperform bonds long-term.", "category": "finance"},
    {"doc_id": "d4", "title": "Central banking",  "body": "Central banks set interest rates.", "category": "economics"},
    {"doc_id": "d5", "title": "Quantitative easing", "body": "QE expands the money supply.", "category": "economics"},
]
