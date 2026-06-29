"""Tests for prism.api.integrations.langgraph — node invocation and state routing."""

import numpy as np
import pytest

from prism.api.consumer import PrismAPIClient
from prism.api.integrations.langgraph import (
    MultiProviderRetrieverNode,
    PrismRetrieverNode,
    create_retriever_node,
)
from prism.api.provider import PrismAPIProvider
from prism.api.schema import APIResponse


DOCS = [
    {"doc_id": "d1", "title": "Inflation", "body": "Rising prices erode value.", "cat": "econ"},
    {"doc_id": "d2", "title": "Bonds",     "body": "Fixed income instruments.",  "cat": "fin"},
    {"doc_id": "d3", "title": "Equities",  "body": "Stock market returns.",       "cat": "fin"},
]


def make_client(projector, embedder, docs=DOCS):
    provider = PrismAPIProvider(
        projector=projector, embedder=embedder,
        semantic_fields=["title", "body"], id_field="doc_id", exact_fields=["cat"],
    )

    @provider.expose
    def search(query: str, top_k: int = 5):
        return docs[:top_k]

    return PrismAPIClient(projector=projector, embedder=embedder, loopback_provider=provider)


class TestPrismRetrieverNode:
    def test_basic_invocation(self, projector, embedder):
        client = make_client(projector, embedder)
        node = PrismRetrieverNode(client=client, query_key="query", results_key="results")
        state = {"query": "what is inflation?"}
        out = node(state)
        assert "results" in out
        assert isinstance(out["results"], APIResponse)

    def test_preserves_unrelated_state_keys(self, projector, embedder):
        client = make_client(projector, embedder)
        node = create_retriever_node(client, results_key="docs")
        state = {"query": "bonds", "session_id": "abc123", "user": "alice"}
        out = node(state)
        # Node only returns the keys it owns; caller merges with graph state
        assert "docs" in out
        assert "session_id" not in out   # node doesn't echo unrelated keys

    def test_query_aliases(self, projector, embedder):
        client = make_client(projector, embedder)
        node = PrismRetrieverNode(client=client)
        # "input" is a common LangGraph alias for "query"
        out = node({"input": "interest rates"})
        assert "search_results" in out

    def test_top_k_override_from_state(self, projector, embedder):
        client = make_client(projector, embedder)
        node = PrismRetrieverNode(
            client=client, top_k=5, top_k_key="top_k_override"
        )
        out = node({"query": "macro", "top_k_override": 2})
        assert len(out["search_results"].results) == 2

    def test_error_key_on_failure(self, projector, embedder):
        # Create a client that always fails
        from unittest.mock import MagicMock
        bad_client = MagicMock()
        bad_client.query.side_effect = ConnectionError("server down")

        node = PrismRetrieverNode(
            client=bad_client,
            error_key="retrieval_error",
        )
        out = node({"query": "test"})
        assert out["retrieval_error"] is not None
        assert "server down" in out["retrieval_error"]
        assert out["search_results"] is None

    def test_fallback_fn_called_on_failure(self, projector, embedder):
        from unittest.mock import MagicMock

        bad_client = MagicMock()
        bad_client.query.side_effect = ConnectionError("server down")

        fallback_results = [{"id": "fb1", "body": "fallback content", "cat": "misc"}]

        node = PrismRetrieverNode(
            client=bad_client,
            fallback_fn=lambda q, k: fallback_results,
            error_key="err",
        )
        out = node({"query": "test"})
        assert out["search_results"] is not None
        assert len(out["search_results"].results) == 1
        assert "fallback" in out["err"]

    def test_empty_query_returns_empty(self, projector, embedder):
        client = make_client(projector, embedder)
        node = PrismRetrieverNode(client=client)
        # query_key missing → empty string query still runs (provider decides)
        out = node({})
        assert "search_results" in out


class TestMultiProviderRetrieverNode:
    def test_fans_out_to_all_providers(self, projector, embedder):
        docs_a = [{"doc_id": "a1", "title": "Alpha", "body": "first corpus", "cat": "a"}]
        docs_b = [{"doc_id": "b1", "title": "Beta",  "body": "second corpus", "cat": "b"}]
        clients = {
            "alpha": make_client(projector, embedder, docs_a),
            "beta":  make_client(projector, embedder, docs_b),
        }
        node = MultiProviderRetrieverNode(clients=clients, top_k_per_provider=1, total_top_k=2)
        out = node({"query": "content"})
        result = out["search_results"]
        assert len(result.results) == 2
        providers_seen = {s["provider"] for s in result.sidecars}
        assert providers_seen == {"alpha", "beta"}


class TestCreateRetrieverNode:
    def test_factory_returns_node(self, projector, embedder):
        client = make_client(projector, embedder)
        node = create_retriever_node(client, top_k=2)
        assert callable(node)
        out = node({"query": "inflation"})
        assert "search_results" in out
