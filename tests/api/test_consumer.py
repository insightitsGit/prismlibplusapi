"""Tests for PrismAPIClient — loopback mode, retry config, query interface."""

import numpy as np
import pytest

from prism.api.consumer import PrismAPIClient, RetryConfig
from prism.api.provider import PrismAPIProvider
from prism.api.schema import APIResponse


def make_provider(projector, embedder, docs):
    provider = PrismAPIProvider(
        projector=projector,
        embedder=embedder,
        semantic_fields=["title", "body"],
        id_field="doc_id",
        exact_fields=["category"],
    )

    @provider.expose
    def search(query: str, top_k: int = 5):
        return docs[:top_k]

    return provider


class TestRetryConfig:
    def test_defaults(self):
        rc = RetryConfig()
        assert rc.max_retries == 3
        assert rc.backoff_base == 0.5
        assert rc.timeout_connect == 5.0

    def test_total_timeout(self):
        rc = RetryConfig(timeout_connect=5.0, timeout_read=30.0)
        assert rc.timeout == 35.0


class TestPrismAPIClientLoopback:
    def test_query_returns_response(self, projector, embedder, SAMPLE_DOCS):
        provider = make_provider(projector, embedder, SAMPLE_DOCS)
        client = PrismAPIClient(
            projector=projector,
            embedder=embedder,
            loopback_provider=provider,
        )
        resp = client.query("inflation and bonds", top_k=3)
        assert isinstance(resp, APIResponse)
        assert len(resp.results) == 3

    def test_vectors_correct_dim(self, projector, embedder, SAMPLE_DOCS):
        from tests.api.conftest import TARGET_DIM
        provider = make_provider(projector, embedder, SAMPLE_DOCS)
        client = PrismAPIClient(
            projector=projector, embedder=embedder, loopback_provider=provider
        )
        resp = client.query("interest rates", top_k=2)
        assert resp.vectors.shape == (2, TARGET_DIM)

    def test_embedding_calls_saved(self, projector, embedder, SAMPLE_DOCS):
        provider = make_provider(projector, embedder, SAMPLE_DOCS)
        client = PrismAPIClient(
            projector=projector, embedder=embedder, loopback_provider=provider
        )
        resp = client.query("bonds", top_k=4)
        # PrismAPI saves one embed call per result returned
        assert resp.embedding_calls_saved == 4

    def test_query_vector_zero_embeds(self, projector, embedder, SAMPLE_DOCS):
        provider = make_provider(projector, embedder, SAMPLE_DOCS)
        client = PrismAPIClient(
            projector=projector, embedder=embedder, loopback_provider=provider
        )
        from tests.api.conftest import TARGET_DIM
        vec = np.random.default_rng(42).standard_normal(TARGET_DIM).astype(np.float32)
        resp = client.query_vector(vec, top_k=2)
        assert isinstance(resp, APIResponse)
        assert len(resp.results) == 2

    def test_sidecars_have_category(self, projector, embedder, SAMPLE_DOCS):
        provider = make_provider(projector, embedder, SAMPLE_DOCS)
        client = PrismAPIClient(
            projector=projector, embedder=embedder, loopback_provider=provider
        )
        resp = client.query("macro", top_k=2)
        for s in resp.sidecars:
            assert "category" in s

    def test_context_manager(self, projector, embedder, SAMPLE_DOCS):
        provider = make_provider(projector, embedder, SAMPLE_DOCS)
        with PrismAPIClient(
            projector=projector, embedder=embedder, loopback_provider=provider
        ) as client:
            resp = client.query("QE", top_k=1)
            assert len(resp.results) == 1


@pytest.fixture
def SAMPLE_DOCS():
    from tests.api.conftest import SAMPLE_DOCS
    return SAMPLE_DOCS
