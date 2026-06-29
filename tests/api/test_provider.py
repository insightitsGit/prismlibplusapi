"""Tests for PrismAPIProvider — projection pipeline and @expose decorator."""

import numpy as np
import pytest

from prism.api.provider import PrismAPIProvider
from prism.api.schema import APIResponse


class TestPrismAPIProvider:
    def test_project_results_returns_response(self, projector, embedder, SAMPLE_DOCS):
        provider = PrismAPIProvider(
            projector=projector,
            embedder=embedder,
            semantic_fields=["title", "body"],
            id_field="doc_id",
            exact_fields=["category"],
        )
        resp = provider.project_results(SAMPLE_DOCS)
        assert isinstance(resp, APIResponse)
        assert len(resp.results) == len(SAMPLE_DOCS)

    def test_vectors_have_correct_dim(self, projector, embedder, SAMPLE_DOCS):
        from tests.api.conftest import TARGET_DIM
        provider = PrismAPIProvider(
            projector=projector,
            embedder=embedder,
            semantic_fields=["title", "body"],
            id_field="doc_id",
        )
        resp = provider.project_results(SAMPLE_DOCS)
        assert resp.vectors.shape == (len(SAMPLE_DOCS), TARGET_DIM)

    def test_exact_fields_not_in_semantic(self, projector, embedder, SAMPLE_DOCS):
        provider = PrismAPIProvider(
            projector=projector,
            embedder=embedder,
            semantic_fields=["title", "body"],
            id_field="doc_id",
            exact_fields=["category"],
        )
        resp = provider.project_results(SAMPLE_DOCS)
        for _, side in resp.results:
            assert "category" in side.fields
            # price is not in SAMPLE_DOCS so absent — no crash expected

    def test_expose_decorator_preserves_callable(self, projector, embedder):
        provider = PrismAPIProvider(
            projector=projector,
            embedder=embedder,
            semantic_fields=["title", "body"],
            id_field="doc_id",
        )

        @provider.expose
        def search(query: str, top_k: int = 5):
            return [{"doc_id": "d1", "title": "test", "body": query}]

        result = search(query="inflation", top_k=1)
        assert isinstance(result, list)
        assert result[0]["body"] == "inflation"

    def test_expose_as_chorus_frame(self, projector, embedder):
        from prism.lib.fabric import CHORUSFrame, FrameType

        provider = PrismAPIProvider(
            projector=projector,
            embedder=embedder,
            semantic_fields=["title", "body"],
            id_field="doc_id",
        )

        @provider.expose
        def search(query: str, top_k: int = 5):
            return [{"doc_id": "d1", "title": "macro", "body": "Fed policy"}]

        frame = search.as_chorus_frame(query="Fed", top_k=1)
        assert isinstance(frame, CHORUSFrame)
        assert frame.frame_type == FrameType.API_RESPONSE

    def test_empty_results(self, projector, embedder):
        provider = PrismAPIProvider(
            projector=projector, embedder=embedder,
            semantic_fields=["title", "body"], id_field="doc_id",
        )
        resp = provider.project_results([])
        assert resp.results == []


@pytest.fixture
def SAMPLE_DOCS():
    from tests.api.conftest import SAMPLE_DOCS
    return SAMPLE_DOCS
