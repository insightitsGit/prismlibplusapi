"""
prism.api.consumer — PrismAPIClient and LangGraphTool
======================================================

The consumer side of PrismAPI.  An agent (e.g. in LangGraph) uses
PrismAPIClient instead of a plain HTTP client to query a PrismAPIProvider
endpoint.  The difference:

    Plain HTTP client:
        agent → HTTP GET /search?q=... → JSON body with text
        agent → embed(text) → float32 vector
        agent → retrieval step with vector
        Cost: 1 HTTP call + 1 embedding call per search

    PrismAPIClient:
        agent → CHORUS API_REQUEST (query vector) → API_RESPONSE (float32 vectors)
        agent → retrieval step with vectors (already embedded + projected)
        Cost: 1 CHORUS frame round-trip, 0 embedding calls on the consumer side

The embedding call is not eliminated globally — the PROVIDER embeds the
content once when it indexes it.  What is eliminated is the CONSUMER re-
embedding on every retrieval.  At scale, a consumer that handles 1,000 queries
per second and gets 10 results per query avoids 10,000 embedding API calls
per second.

In-process vs networked
-----------------------
PrismAPIClient works in two modes:

    Networked (production):
        Pass `host` and `port` pointing at a server running ASGIAdapter.
        Frames travel over HTTP(S) with Content-Type: application/x-chorus-frame.
        (Full gRPC support is on the roadmap; HTTP transport is sufficient for
        throughput at typical retrieval scales.)

    In-process (loopback):
        Pass a PrismAPIProvider directly as `loopback_provider`.
        Frames are serialised and deserialised in memory — identical wire path,
        no network involved.  Used in benchmarks and tests.
"""

from __future__ import annotations

import http.client
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from prism.lib.fabric import CHORUSFrame, FabricConfig, FrameType, TensorCipher
from prism.lib.lang import PrismProjector, ProjectionConfig
from prism.api.schema import (
    APIRequest,
    APIResponse,
    Embedder,
    ExactSidecar,
    SemanticItem,
    unpack_response_payload,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """
    Controls retry behaviour for the production HTTP client.

    Attributes
    ----------
    max_retries:
        Maximum number of retry attempts after the initial failure (0 = no retry).
    backoff_base:
        Base sleep duration in seconds.  Sleep doubles each retry:
        backoff_base, 2×, 4× ...  Capped at backoff_max.
    backoff_max:
        Maximum sleep between retries.
    timeout_connect:
        TCP connect timeout in seconds.
    timeout_read:
        Socket read timeout in seconds.
    """

    max_retries: int = 3
    backoff_base: float = 0.5
    backoff_max: float = 8.0
    timeout_connect: float = 5.0
    timeout_read: float = 30.0

    @property
    def timeout(self) -> float:
        """Total socket timeout (connect + read)."""
        return self.timeout_connect + self.timeout_read


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConsumerError(Exception):
    """Base error for PrismAPIClient operations."""


class FrameTypeError(ConsumerError):
    """Raised when the server returns an unexpected frame type."""


class TransportError(ConsumerError):
    """Raised on network-level failures."""


# ---------------------------------------------------------------------------
# PrismAPIClient
# ---------------------------------------------------------------------------


class PrismAPIClient:
    """
    Connects to a PrismAPIProvider endpoint and retrieves float32 vectors
    plus exact sidecar metadata — no re-embedding on the consumer side.

    Parameters
    ----------
    projector:
        PrismProjector for the consumer's tenant.  Used to project the
        query embedding into the provider's target_dim space before sending.
    embedder:
        Embedder used for the query text.  The consumer still embeds its
        OWN query — what it avoids is embedding the RESULTS it gets back.
    loopback_provider:
        If provided, frames are exchanged in-process (for tests / benchmarks).
        Mutually exclusive with host/port.
    host, port:
        Remote PrismAPIProvider address for networked mode.
    source_field:
        Label to attach to returned SemanticItems (informational only).
    """

    def __init__(
        self,
        projector: PrismProjector,
        embedder: Embedder,
        loopback_provider: Optional[Any] = None,   # PrismAPIProvider
        host: str = "localhost",
        port: int = 9100,
        source_field: str = "body",
        retry: Optional[RetryConfig] = None,
        chorus_path: str = "/chorus/search",
        api_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
    ) -> None:
        self._projector = projector
        self._embedder = embedder
        self._loopback = loopback_provider
        self._host = host
        self._port = port
        self._source_field = source_field
        self._retry = retry or RetryConfig()
        self._chorus_path = chorus_path
        self._api_key = api_key
        self._bearer_token = bearer_token

        # Persistent HTTP connection (keep-alive, reused across requests)
        self._conn: Optional[http.client.HTTPConnection] = None

        # Cipher for signing outbound request frames
        dim = projector._cfg.target_dim
        self._cipher = TensorCipher(dim=dim, ttl_seconds=3600.0)
        self._cipher.rotate_key()
        self._seq = 0

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> http.client.HTTPConnection:
        """Return a live persistent connection, creating one if needed."""
        if self._conn is None:
            self._conn = http.client.HTTPConnection(
                self._host,
                self._port,
                timeout=self._retry.timeout,
            )
        return self._conn

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def close(self) -> None:
        """Close the persistent HTTP connection."""
        self._close_conn()

    def __enter__(self) -> "PrismAPIClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def health_check(self) -> bool:
        """
        Ping the server's /health endpoint.

        Returns True if the server responds 200 OK, False otherwise.
        """
        try:
            conn = self._get_conn()
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            return resp.status == 200
        except Exception:
            self._close_conn()
            return False

    # ------------------------------------------------------------------
    # Main query interface
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> APIResponse:
        """
        Embed ``query_text``, send as a CHORUS API_REQUEST, return an
        APIResponse with pre-projected float32 vectors.

        The consumer embeds the query once.  The provider returns already-
        projected result vectors — no second embed() call is made here.

        Parameters
        ----------
        query_text:
            The agent's natural-language query.
        top_k:
            Number of results to request.
        extra_context:
            Additional query parameters forwarded to the provider.

        Returns
        -------
        APIResponse — results are (SemanticItem, ExactSidecar) pairs.
        Use response.vectors for a stacked (N, dim) array ready for
        PrismResonance retrieval.
        """
        from prism.observability.otel import trace_span

        with trace_span(
            "prism.api.client.query",
            host=self._host,
            port=self._port,
            loopback=self._loopback is not None,
        ):
            t0 = time.perf_counter()

            # Embed query (one call — the only embedding on the consumer side)
            raw_emb = self._embedder.embed([query_text])[0]   # (embed_dim,)
            envelope = self._projector.project(raw_emb)
            query_vec = envelope.vector   # (target_dim,) float32

            context: dict[str, Any] = {
                "query_text": query_text,
                "top_k": top_k,
                **(extra_context or {}),
            }
            request = APIRequest(query_vector=query_vec, context=context)

            # Exchange frames
            resp_frame = self._exchange(request)

            # Decode response
            raw_results = resp_frame.decode_api_response()
            result_pairs = unpack_response_payload(raw_results, source_field=self._source_field)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return APIResponse(
                results=result_pairs,
                provider_id="remote" if self._loopback is None else self._loopback._provider_id,
                request_id=request.request_id,
                latency_ms=latency_ms,
                embedding_calls_saved=len(result_pairs),
            )

    def query_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> APIResponse:
        """
        Send a pre-computed query vector (already in target_dim space)
        without any embedding step.

        Use this when the agent already has a vector (e.g. from a previous
        retrieval step) and wants to find similar content from the provider.
        Zero embedding calls.
        """
        t0 = time.perf_counter()

        context: dict[str, Any] = {"top_k": top_k, **(extra_context or {})}
        request = APIRequest(query_vector=query_vector, context=context)
        resp_frame = self._exchange(request)
        raw_results = resp_frame.decode_api_response()
        result_pairs = unpack_response_payload(raw_results, source_field=self._source_field)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return APIResponse(
            results=result_pairs,
            provider_id="remote" if self._loopback is None else self._loopback._provider_id,
            request_id=request.request_id,
            latency_ms=latency_ms,
            embedding_calls_saved=len(result_pairs),
        )

    # ------------------------------------------------------------------
    # Frame exchange
    # ------------------------------------------------------------------

    def _exchange(self, request: APIRequest) -> CHORUSFrame:
        """
        Serialise the request as a CHORUSFrame and return the response frame.

        In loopback mode: in-process frame serialisation → provider → frame bytes.
        In networked mode: HTTP POST with Content-Type: application/x-chorus-frame.
        """
        key = self._cipher._active_key  # type: ignore[union-attr]

        # Build request frame
        req_frame = CHORUSFrame.from_api_request(
            key_id=key.key_id,
            seq=self._seq,
            watermark=b"\x00" * 32,   # consumer watermark — provider verifies its own
            query_vector=request.query_vector,
            context=request.context,
        )
        self._seq += 1

        if self._loopback is not None:
            return self._loopback_exchange(req_frame, request.context)
        return self._http_exchange(req_frame)

    def _loopback_exchange(
        self, req_frame: CHORUSFrame, context: dict[str, Any]
    ) -> CHORUSFrame:
        """In-process exchange: serialise → provider.handle → deserialise."""
        # Wire path: to_bytes() → from_bytes() exercises full serialisation
        wire_bytes = req_frame.to_bytes()
        decoded_req = CHORUSFrame.from_bytes(wire_bytes)
        query_vec, ctx = decoded_req.decode_api_request()

        query_text = ctx.get("query_text", "")
        top_k = int(ctx.get("top_k", 10))

        # Invoke the loopback provider directly
        result_dicts = self._loopback._invoke_handler(query_text, query_vec, top_k)
        resp_frame = self._loopback.as_chorus_frame(result_dicts)

        # Round-trip through bytes to include wire overhead in timing
        return CHORUSFrame.from_bytes(resp_frame.to_bytes())

    def _http_exchange(self, req_frame: CHORUSFrame) -> CHORUSFrame:
        """
        Networked exchange over HTTP with retry and persistent connection.

        Uses http.client.HTTPConnection with keep-alive for connection reuse.
        Retries on transient errors (ConnectionError, timeout) with exponential
        backoff.  Re-establishes connection after any transport failure.
        """
        data = req_frame.to_bytes()
        headers = {
            "Content-Type": "application/x-chorus-frame",
            "Content-Length": str(len(data)),
            "Connection": "keep-alive",
        }
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"

        last_exc: Optional[Exception] = None
        for attempt in range(self._retry.max_retries + 1):
            if attempt > 0:
                sleep_s = min(
                    self._retry.backoff_base * (2 ** (attempt - 1)),
                    self._retry.backoff_max,
                )
                logger.warning(
                    "PrismAPIClient: retry %d/%d after %.1fs (error: %s)",
                    attempt,
                    self._retry.max_retries,
                    sleep_s,
                    last_exc,
                )
                time.sleep(sleep_s)

            try:
                conn = self._get_conn()
                conn.request("POST", self._chorus_path, body=data, headers=headers)
                resp = conn.getresponse()
                body = resp.read()

                if resp.status != 200:
                    raise TransportError(
                        f"Server returned HTTP {resp.status}: {body[:200]}"
                    )

                frame = CHORUSFrame.from_bytes(body)
                if frame.frame_type != FrameType.API_RESPONSE:
                    raise FrameTypeError(
                        f"Expected API_RESPONSE, got {frame.frame_type.name}"
                    )
                return frame

            except (FrameTypeError, TransportError):
                # Non-retryable protocol errors — surface immediately
                raise
            except Exception as exc:
                last_exc = exc
                # Connection-level error — drop and reconnect on next attempt
                self._close_conn()

        raise TransportError(
            f"HTTP exchange failed after {self._retry.max_retries + 1} attempts: {last_exc}"
        ) from last_exc


# ---------------------------------------------------------------------------
# LangGraphTool — thin adapter for LangGraph agent nodes
# ---------------------------------------------------------------------------


class LangGraphTool:
    """
    Exposes PrismAPIClient as a LangGraph-compatible tool node.

    The agent calls the tool by name; the tool queries the PrismAPIProvider
    and returns a dict with ``vectors`` (np.ndarray) and ``results``
    (list of sidecar dicts) that the agent can use directly.

    Usage in a LangGraph graph::

        from prism.api.consumer import LangGraphTool

        tool = LangGraphTool(
            name="semantic_search",
            description="Search the knowledge base by semantic meaning.",
            client=my_prism_client,
        )

        # In a LangGraph node:
        result = tool.invoke({"query": "how does inflation affect bonds?"})
        # result["vectors"]  — np.ndarray (N, 64), ready for PrismResonance
        # result["sidecars"] — list of exact metadata dicts
        # result["top_k"]    — int, number of results

    LangGraph integration (optional import)::

        # If langgraph is installed, tool.as_langgraph_node() returns a node
        # function compatible with StateGraph.add_node().
        node = tool.as_langgraph_node()
    """

    def __init__(
        self,
        name: str,
        description: str,
        client: PrismAPIClient,
        top_k: int = 10,
    ) -> None:
        self.name = name
        self.description = description
        self._client = client
        self._top_k = top_k

    def invoke(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Synchronous tool invocation.

        Parameters
        ----------
        input_dict:
            Must contain "query" (str).  Optional: "top_k" (int).

        Returns
        -------
        dict with keys:
            "vectors"  — np.ndarray (N, dim), stacked result vectors
            "sidecars" — list[dict], exact metadata for each result
            "top_k"    — int, actual number of results returned
            "latency_ms" — float, end-to-end latency
        """
        query = str(input_dict.get("query", ""))
        top_k = int(input_dict.get("top_k", self._top_k))
        if not query:
            return {"vectors": np.empty((0,), np.float32), "sidecars": [], "top_k": 0}

        response = self._client.query(query, top_k=top_k)
        return {
            "vectors": response.vectors,
            "sidecars": response.sidecars,
            "top_k": len(response.results),
            "latency_ms": response.latency_ms,
            "embedding_calls_saved": response.embedding_calls_saved,
        }

    def as_langgraph_node(self) -> Callable:
        """
        Return a LangGraph node function.

        Requires: pip install langgraph
        The returned function signature is ``node(state: dict) -> dict``,
        compatible with ``StateGraph.add_node(name, node)``.
        """
        try:
            from langgraph.graph import StateGraph  # type: ignore[import]  # noqa: F401
        except ImportError:
            logger.warning(
                "langgraph not installed — as_langgraph_node() returns a plain "
                "callable usable as a node function, but graph registration "
                "requires: pip install langgraph"
            )

        tool = self

        def node(state: dict[str, Any]) -> dict[str, Any]:
            query = state.get("query", state.get("input", ""))
            result = tool.invoke({"query": query})
            return {**state, "prismapi_result": result}

        node.__name__ = self.name
        return node

    # ------------------------------------------------------------------
    # Tool schema for MCP / OpenAI function-calling
    # ------------------------------------------------------------------

    @property
    def tool_schema(self) -> dict[str, Any]:
        """JSON schema describing this tool for MCP or function-calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return.",
                        "default": self._top_k,
                    },
                },
                "required": ["query"],
            },
        }
