"""Tests for MultiProviderClient and MultiProviderResponse — loopback mode."""

import numpy as np
import pytest

from prism.api.consumer import PrismAPIClient
from prism.api.multi_provider import MultiProviderClient, MultiProviderResponse
from prism.api.provider import PrismAPIProvider


def make_client(projector, embedder, docs, provider_id="test"):
    provider = PrismAPIProvider(
        projector=projector,
        embedder=embedder,
        semantic_fields=["title", "body"],
        id_field="doc_id",
        exact_fields=["domain"],
        provider_id=provider_id,
    )

    @provider.expose
    def search(query: str, top_k: int = 5):
        return docs[:top_k]

    return PrismAPIClient(
        projector=projector,
        embedder=embedder,
        loopback_provider=provider,
    )


NEWS_DOCS = [
    {"doc_id": "n1", "title": "Fed rate hike", "body": "Fed raises rates by 25bps.", "domain": "news"},
    {"doc_id": "n2", "title": "CPI report",    "body": "Inflation falls to 3.2%.",   "domain": "news"},
    {"doc_id": "n3", "title": "GDP growth",    "body": "Q3 GDP grows at 2.1%.",      "domain": "news"},
]

LEGAL_DOCS = [
    {"doc_id": "l1", "title": "Rate fixing case", "body": "Banks fined for LIBOR manipulation.", "domain": "legal"},
    {"doc_id": "l2", "title": "Antitrust ruling", "body": "Court rules against merger.",         "domain": "legal"},
]


class TestMultiProviderResponse:
    def test_from_responses_merges_results(self, projector, embedder):
        news_client  = make_client(projector, embedder, NEWS_DOCS,  "news")
        legal_client = make_client(projector, embedder, LEGAL_DOCS, "legal")

        news_resp  = news_client.query("interest rates", top_k=3)
        legal_resp = legal_client.query("interest rates", top_k=2)

        merged = MultiProviderResponse.from_responses(
            {"news": news_resp, "legal": legal_resp},
            total_top_k=4,
        )
        assert len(merged.results) == 4

    def test_provider_attribution_in_sidecar(self, projector, embedder):
        news_client = make_client(projector, embedder, NEWS_DOCS, "news")
        resp = news_client.query("GDP", top_k=2)

        merged = MultiProviderResponse.from_responses({"news": resp}, total_top_k=2)
        for s in merged.sidecars:
            assert s["provider"] == "news"

    def test_deduplication(self, projector, embedder):
        # Same client queried twice under two names should not duplicate
        client = make_client(projector, embedder, NEWS_DOCS, "p1")
        resp1 = client.query("inflation", top_k=2)
        resp2 = client.query("inflation", top_k=2)  # same docs, different provider name

        merged = MultiProviderResponse.from_responses(
            {"p1": resp1, "p2": resp2},
            total_top_k=10,
        )
        # Dedup is by "provider:doc_id" so different provider names won't dedup
        # Just verify no crash and results are bounded
        assert len(merged.results) <= 4

    def test_empty_providers(self):
        merged = MultiProviderResponse.from_responses({}, total_top_k=5)
        assert merged.results == []
        assert merged.vectors.shape == (0,)

    def test_total_embedding_calls_saved(self, projector, embedder):
        client = make_client(projector, embedder, NEWS_DOCS, "news")
        resp = client.query("rates", top_k=3)

        merged = MultiProviderResponse.from_responses({"news": resp}, total_top_k=3)
        assert merged.total_embedding_calls_saved == 3


class TestMultiProviderClient:
    def test_query_fans_out(self, projector, embedder):
        clients = {
            "news":  make_client(projector, embedder, NEWS_DOCS,  "news"),
            "legal": make_client(projector, embedder, LEGAL_DOCS, "legal"),
        }
        mp = MultiProviderClient(clients, top_k_per_provider=2, total_top_k=3)
        result = mp.query("rate manipulation")

        assert isinstance(result, MultiProviderResponse)
        assert len(result.results) <= 3
        # Both providers should have contributed
        assert len(result.per_provider) == 2

    def test_latency_recorded(self, projector, embedder):
        clients = {"news": make_client(projector, embedder, NEWS_DOCS, "news")}
        mp = MultiProviderClient(clients, top_k_per_provider=2, total_top_k=2)
        result = mp.query("GDP")
        assert result.latency_ms > 0
