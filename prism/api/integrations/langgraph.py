"""
prism.api.integrations.langgraph — Production LangGraph nodes
=============================================================

Drop-in retrieval nodes for LangGraph StateGraph pipelines.
Eliminates re-embedding of retrieved results at every agent hop.

Quick start::

    from prism.api.integrations.langgraph import create_retriever_node

    graph = StateGraph(MyState)
    graph.add_node("retrieve", create_retriever_node(
        client=my_prism_client,
        query_key="query",
        results_key="search_results",
        top_k=10,
    ))

Multi-provider fan-out::

    from prism.api.integrations.langgraph import MultiProviderRetrieverNode

    node = MultiProviderRetrieverNode(
        clients={"news": news_client, "legal": legal_client, "wiki": wiki_client},
        query_key="query",
        results_key="search_results",
        top_k_per_provider=5,
    )
    graph.add_node("retrieve", node)
    # Queries all 3 providers in parallel, merges by cosine score, returns top 10

Architecture note
-----------------
In a standard LangGraph pipeline with HTTP/REST retrieval:

    Agent A → retrieve → JSON text → [embed × top_k] → Agent B → reason
                                      ^^^^^^^^^^^^^^^^^^^
                                      this tax repeats at every hop

With PrismRetrieverNode:

    Agent A → retrieve → float32 vectors → Agent B → reason
                                            (no embed call)

Vectors propagate through the state dict between nodes.  Any downstream
node receives pre-projected float32 vectors it can use directly for
cosine ranking, clustering, or PrismResonance retrieval.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PrismRetrieverNode — single-provider, sync + async
# ---------------------------------------------------------------------------


class PrismRetrieverNode:
    """
    LangGraph node that retrieves from a single PrismAPIProvider endpoint.

    Parameters
    ----------
    client:
        PrismAPIClient instance (networked or loopback).
    query_key:
        State dict key to read the query string from.
    results_key:
        State dict key to write the APIResponse under.
    top_k:
        Number of results to request.  Can be overridden per-invocation
        by setting state[top_k_key] (see top_k_key param).
    top_k_key:
        Optional state key to read a dynamic top_k from.  If present and
        the key exists in state, overrides the default top_k.
    error_key:
        If set, write any retrieval error message here instead of raising.
        Enables graceful degradation: downstream nodes check this key.
    fallback_fn:
        Optional callable(query: str, top_k: int) → list[dict] invoked when
        the CHORUS endpoint fails.  Returns plain dicts; the node wraps them
        in an empty-vector APIResponse so downstream nodes see a consistent
        interface.

    Usage in a StateGraph::

        from prism.api.integrations.langgraph import PrismRetrieverNode

        class MyState(TypedDict):
            query: str
            search_results: Optional[APIResponse]
            error: Optional[str]

        node = PrismRetrieverNode(
            client=my_client,
            query_key="query",
            results_key="search_results",
            error_key="error",
            top_k=10,
        )

        graph = StateGraph(MyState)
        graph.add_node("retrieve", node)

    Async usage (LangGraph async graphs)::

        # Just use the node normally — LangGraph calls __call__ or ainvoke
        # depending on whether the graph is compiled with async=True.
        # PrismRetrieverNode implements both.
    """

    def __init__(
        self,
        client: Any,                          # PrismAPIClient
        query_key: str = "query",
        results_key: str = "search_results",
        top_k: int = 10,
        top_k_key: Optional[str] = None,
        error_key: Optional[str] = None,
        fallback_fn: Optional[Callable] = None,
    ) -> None:
        self._client = client
        self._query_key = query_key
        self._results_key = results_key
        self._top_k = top_k
        self._top_k_key = top_k_key
        self._error_key = error_key
        self._fallback_fn = fallback_fn

    # ------------------------------------------------------------------
    # Sync invocation — LangGraph sync graphs
    # ------------------------------------------------------------------

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        query = self._extract_query(state)
        top_k = self._extract_top_k(state)

        try:
            response = self._client.query(query, top_k=top_k)
            return self._success(state, response)
        except Exception as exc:
            return self._handle_error(state, query, top_k, exc)

    # ------------------------------------------------------------------
    # Async invocation — LangGraph async graphs
    # ------------------------------------------------------------------

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Async variant.  The underlying HTTP call is synchronous (urllib),
        so we run it in a thread pool to avoid blocking the event loop.
        """
        query = self._extract_query(state)
        top_k = self._extract_top_k(state)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self._client.query(query, top_k=top_k)
            )
            return self._success(state, response)
        except Exception as exc:
            return self._handle_error(state, query, top_k, exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_query(self, state: dict[str, Any]) -> str:
        query = state.get(self._query_key, "")
        if not query:
            # Common aliases — be forgiving about state schema
            for alias in ("input", "question", "user_query", "search_query"):
                if state.get(alias):
                    return str(state[alias])
        return str(query)

    def _extract_top_k(self, state: dict[str, Any]) -> int:
        if self._top_k_key and self._top_k_key in state:
            try:
                return int(state[self._top_k_key])
            except (TypeError, ValueError):
                pass
        return self._top_k

    def _success(self, state: dict[str, Any], response: Any) -> dict[str, Any]:
        update = {self._results_key: response}
        if self._error_key:
            update[self._error_key] = None
        logger.debug(
            "PrismRetrieverNode: %d results, %.1f ms, %d embed calls saved",
            len(response.results),
            response.latency_ms,
            response.embedding_calls_saved,
        )
        return update

    def _handle_error(
        self,
        state: dict[str, Any],
        query: str,
        top_k: int,
        exc: Exception,
    ) -> dict[str, Any]:
        msg = f"PrismRetrieverNode retrieval failed: {exc}"
        logger.error(msg)

        if self._fallback_fn is not None:
            try:
                fallback_results = self._fallback_fn(query, top_k)
                # Wrap in a minimal APIResponse-like object so downstream
                # nodes see a consistent interface
                from prism.api.schema import APIResponse, ExactSidecar, SemanticItem
                pairs = [
                    (
                        SemanticItem(
                            doc_id=str(r.get("id", i)),
                            vector=np.zeros(64, dtype=np.float32),
                            source_field="fallback",
                            text_preview=str(r.get("body", r.get("text", "")))[:120],
                        ),
                        ExactSidecar(doc_id=str(r.get("id", i)), fields=r),
                    )
                    for i, r in enumerate(fallback_results)
                ]
                import uuid
                response = APIResponse(
                    results=pairs,
                    provider_id="fallback",
                    request_id=str(uuid.uuid4()),
                    embedding_calls_saved=0,
                )
                update = {self._results_key: response}
                if self._error_key:
                    update[self._error_key] = f"fallback: {msg}"
                return update
            except Exception as fb_exc:
                logger.error("Fallback also failed: %s", fb_exc)

        if self._error_key:
            return {self._results_key: None, self._error_key: msg}

        raise


# ---------------------------------------------------------------------------
# MultiProviderRetrieverNode — parallel fan-out across N providers
# ---------------------------------------------------------------------------


class MultiProviderRetrieverNode:
    """
    LangGraph node that queries multiple PrismAPIProviders in parallel and
    merges results by cosine score in the shared projected space.

    All providers MUST use the same tenant_id so their vectors are in the
    same projected space and are directly comparable.

    Parameters
    ----------
    clients:
        Dict mapping provider name → PrismAPIClient.
        E.g. {"news": news_client, "legal": legal_client}
    query_key:
        State key to read query from.
    results_key:
        State key to write merged MultiProviderResponse to.
    top_k_per_provider:
        How many results to fetch from each provider.
    total_top_k:
        How many results to return after merging.  Defaults to
        top_k_per_provider (take the best across all providers).
    error_key:
        If set, write per-provider error summary here.

    Why this matters
    ----------------
    Querying 3 providers sequentially over HTTP with re-embedding:
        3 × (RTT + top_k × embed_ms) ≈ 3 × (50 ms + 10 × 30 ms) = 1,050 ms

    MultiProviderRetrieverNode (parallel, no re-embed):
        max(RTT_1, RTT_2, RTT_3) ≈ 50 ms (slowest provider latency)

    Merging is free — all results are in the same float32 projected space.
    """

    def __init__(
        self,
        clients: dict[str, Any],             # name → PrismAPIClient
        query_key: str = "query",
        results_key: str = "search_results",
        top_k_per_provider: int = 5,
        total_top_k: Optional[int] = None,
        error_key: Optional[str] = None,
    ) -> None:
        self._clients = clients
        self._query_key = query_key
        self._results_key = results_key
        self._top_k_per_provider = top_k_per_provider
        self._total_top_k = total_top_k or top_k_per_provider
        self._error_key = error_key

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from prism.api.multi_provider import MultiProviderResponse

        query = str(state.get(self._query_key, ""))
        errors: dict[str, str] = {}
        provider_responses: dict[str, Any] = {}

        def _fetch(name: str, client: Any):
            return name, client.query(query, top_k=self._top_k_per_provider)

        with ThreadPoolExecutor(max_workers=len(self._clients)) as pool:
            futures = {
                pool.submit(_fetch, name, client): name
                for name, client in self._clients.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    pname, response = future.result()
                    provider_responses[pname] = response
                except Exception as exc:
                    errors[name] = str(exc)
                    logger.error("Provider %s failed: %s", name, exc)

        merged = MultiProviderResponse.from_responses(
            provider_responses, total_top_k=self._total_top_k
        )
        update: dict[str, Any] = {self._results_key: merged}
        if self._error_key:
            update[self._error_key] = errors if errors else None
        return update

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.__call__(state))


# ---------------------------------------------------------------------------
# Factory — one-liner for simple cases
# ---------------------------------------------------------------------------


def create_retriever_node(
    client: Any,
    query_key: str = "query",
    results_key: str = "search_results",
    top_k: int = 10,
    error_key: Optional[str] = None,
    fallback_fn: Optional[Callable] = None,
) -> PrismRetrieverNode:
    """
    Factory for the common case.

    Example::

        graph.add_node("retrieve", create_retriever_node(
            client=my_client,
            query_key="query",
            results_key="search_results",
            top_k=10,
        ))
    """
    return PrismRetrieverNode(
        client=client,
        query_key=query_key,
        results_key=results_key,
        top_k=top_k,
        error_key=error_key,
        fallback_fn=fallback_fn,
    )
