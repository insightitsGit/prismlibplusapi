"""
prism.api.provider — PrismAPIProvider and @expose decorator
============================================================

An API owner installs PrismAPIProvider on top of an existing handler with
a single decorator line:

    import prism.api as prismapi

    provider = prismapi.PrismAPIProvider(
        projector=my_projector,
        embedder=SentenceTransformerEmbedder(),
        semantic_fields=["title", "body"],
        id_field="doc_id",
        exact_fields=["price", "url", "in_stock"],   # never vectorized
    )

    @provider.expose
    def search(query: str) -> list[dict]:
        return db.search(query)        # existing handler — unchanged

The decorator wraps `search` so that:
1.  The original handler runs normally (its HTTP/REST interface is unchanged).
2.  For every response dict, `title` and `body` are concatenated, embedded
    via the injected Embedder, and projected to 64-d float32 via PrismProjector.
3.  `price`, `url`, and `in_stock` ride as a plain JSON sidecar.
4.  The result is available as a CHORUSFrame (API_RESPONSE) for CHORUS consumers.

Architecture note on serialisation boundary
-------------------------------------------
The provider-to-handler boundary is plain Python dicts — the handler is never
aware of CHORUS.  Serialisation to CHORUSFrame happens AFTER the handler
returns, inside PrismAPIProvider.  The app-to-bridge boundary eliminates
JSON encoding of the vector payload; the handler-to-db boundary necessarily
speaks whatever protocol the DB requires (this is the same honest note as
in prism.bridge.vector).

ASGI adapter
------------
ASGIAdapter adds a `/chorus/<path>` endpoint next to the existing HTTP routes.
It accepts POST requests with Content-Type: application/x-chorus-frame,
deserialises the API_REQUEST frame, invokes the handler, and returns an
API_RESPONSE frame.  Existing HTTP routes are completely unaffected.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from typing import Any, Callable, Optional, Sequence

import numpy as np

from prism.lib.fabric import (
    CHORUSFrame,
    FabricConfig,
    FrameType,
    TensorCipher,
)
from prism.lib.lang import PayloadEnvelope, PrismProjector, ProjectionConfig
from prism.api.schema import (
    APIResponse,
    Embedder,
    ExactSidecar,
    SemanticItem,
    pack_response_payload,
    unpack_response_payload,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base error for PrismAPIProvider operations."""


class MissingFieldError(ProviderError):
    """Raised when a required field is absent from a handler result dict."""


class EmbedderNotConfiguredError(ProviderError):
    """Raised when no embedder has been injected into the provider."""


# ---------------------------------------------------------------------------
# PrismAPIProvider
# ---------------------------------------------------------------------------


class PrismAPIProvider:
    """
    Wraps an existing handler/function to serve its semantic content as
    CHORUS float32 vectors alongside the unchanged HTTP response.

    Parameters
    ----------
    projector:
        PrismProjector instance for this tenant.  Provides the JL reduction
        and tenant-isolation math (SHA-256(tenant_id) seeded).
    embedder:
        Any object satisfying the Embedder protocol (embed(texts) → ndarray).
        Injected — the provider does not create or own the embedder.
    semantic_fields:
        Field names from handler result dicts whose text content should be
        embedded and projected.  Multiple fields are concatenated with " | ".
    id_field:
        Field name to use as doc_id.  Defaults to "id".
    exact_fields:
        Field names that must NOT be vectorized — they ride as sidecar JSON.
        If empty, all non-semantic, non-id fields are treated as exact.
    provider_id:
        Stable identifier for this provider endpoint.  Reported in responses.
    """

    def __init__(
        self,
        projector: PrismProjector,
        embedder: Embedder,
        semantic_fields: Sequence[str],
        id_field: str = "id",
        exact_fields: Optional[Sequence[str]] = None,
        provider_id: Optional[str] = None,
    ) -> None:
        self._projector = projector
        self._embedder = embedder
        self._semantic_fields = list(semantic_fields)
        self._id_field = id_field
        self._exact_fields = list(exact_fields) if exact_fields is not None else []
        self._provider_id = provider_id or str(uuid.uuid4())

        # Cipher for signing outbound frames
        self._cipher = TensorCipher(
            dim=projector._cfg.target_dim,
            ttl_seconds=3600.0,
        )
        self._cipher.rotate_key()
        self._seq = 0

        logger.info(
            "PrismAPIProvider[%s]: semantic_fields=%s id_field=%s",
            self._provider_id,
            self._semantic_fields,
            self._id_field,
        )

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def expose(self, fn: Callable) -> "ExposedHandler":
        """
        Decorator: wrap an existing handler so it also serves CHORUS frames.

        Usage::

            @provider.expose
            def search(query: str) -> list[dict]:
                return db.search(query)   # unchanged

        The returned ExposedHandler is callable with the same signature as the
        original handler.  It additionally exposes:
            .as_chorus_frame(result_dicts) → CHORUSFrame
            .as_api_response(result_dicts) → APIResponse
        """
        return ExposedHandler(fn, self)

    # ------------------------------------------------------------------
    # Core projection pipeline
    # ------------------------------------------------------------------

    def project_results(
        self,
        result_dicts: list[dict[str, Any]],
    ) -> APIResponse:
        """
        Convert a list of handler result dicts into a full APIResponse.

        Steps:
            1. Extract text from semantic_fields (concatenated with " | ").
            2. Embed all texts in one batch (one embedder call total).
            3. Project each embedding via PrismProjector → 64-d float32.
            4. Build SemanticItem and ExactSidecar for each result.
            5. Wrap in APIResponse.

        The embedder is called ONCE for the entire batch — not once per item.
        At N=10 results with 1 embedding call vs N individual calls, this is
        the primary token/compute saving on the PROVIDER side.  The consumer
        saves its own embedding calls by receiving pre-projected vectors.
        """
        if not result_dicts:
            return APIResponse(
                results=[],
                provider_id=self._provider_id,
                request_id=str(uuid.uuid4()),
                embedding_calls_saved=0,
            )

        t0 = time.perf_counter()

        # --- Step 1: extract semantic text --------------------------------
        texts: list[str] = []
        for item in result_dicts:
            parts = []
            for field in self._semantic_fields:
                val = item.get(field, "")
                if val:
                    parts.append(str(val))
            texts.append(" | ".join(parts) if parts else "")

        # --- Step 2: embed in one batch -----------------------------------
        embeddings = self._embedder.embed(texts)   # (N, embed_dim)

        # --- Step 3: project each embedding ------------------------------
        envelopes: list[PayloadEnvelope] = [
            self._projector.project(embeddings[i])
            for i in range(len(result_dicts))
        ]

        # --- Step 4: build result pairs ----------------------------------
        results: list[tuple[SemanticItem, ExactSidecar]] = []
        for i, (item, env) in enumerate(zip(result_dicts, envelopes)):
            doc_id = str(item.get(self._id_field, f"doc_{i}"))

            # Semantic item
            preview_text = texts[i][:120]
            sem = SemanticItem(
                doc_id=doc_id,
                vector=env.vector,
                source_field=" | ".join(self._semantic_fields),
                text_preview=preview_text,
            )

            # Exact sidecar — all fields NOT in semantic_fields, or explicitly listed
            if self._exact_fields:
                exact_data = {k: item[k] for k in self._exact_fields if k in item}
            else:
                skip = set(self._semantic_fields) | {self._id_field}
                exact_data = {k: v for k, v in item.items() if k not in skip}
            side = ExactSidecar(doc_id=doc_id, fields=exact_data)

            results.append((sem, side))

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return APIResponse(
            results=results,
            provider_id=self._provider_id,
            request_id=str(uuid.uuid4()),
            latency_ms=latency_ms,
            embedding_calls_saved=len(result_dicts),
        )

    def as_chorus_frame(
        self,
        result_dicts: list[dict[str, Any]],
    ) -> CHORUSFrame:
        """
        Project result_dicts and pack them as a signed CHORUSFrame (API_RESPONSE).

        The frame is HMAC-signed using the provider's TensorCipher session key.
        The consumer must hold the same key to verify — in production this is
        negotiated during channel setup; in loopback mode the same cipher
        instance is shared.
        """
        response = self.project_results(result_dicts)
        pairs = pack_response_payload(response.results)

        key = self._cipher._active_key  # type: ignore[union-attr]
        vec_payload = np.stack([v for v, _ in pairs], axis=0).astype(np.float32) if pairs else np.empty((0,), np.float32)

        # Encrypt vector payload, compute watermark
        if pairs:
            V_enc, watermark = self._cipher.encrypt(vec_payload, sequence_number=self._seq)
        else:
            V_enc = vec_payload
            import hmac as _hmac, hashlib
            watermark = _hmac.new(key.stream_secret, b"empty", hashlib.sha256).digest()

        frame = CHORUSFrame.from_api_response(
            key_id=key.key_id,
            seq=self._seq,
            watermark=watermark,
            results=[(v, s) for v, s in pairs],
        )
        self._seq += 1
        return frame

    def _invoke_handler(
        self,
        query_text: str,
        query_vector: "np.ndarray",
        top_k: int,
    ) -> "list[dict]":
        """
        Internal: called by PrismAPIClient in loopback mode.

        Subclasses or ExposedHandler wrappers must set _handler to be callable.
        If no _handler is registered (raw PrismAPIProvider with no @expose),
        this raises ProviderError — callers should use as_chorus_frame() with
        pre-fetched result_dicts instead.
        """
        handler = getattr(self, "_registered_handler", None)
        if handler is None:
            raise ProviderError(
                "No handler registered on this provider. "
                "Use @provider.expose or pass result_dicts directly to as_chorus_frame()."
            )
        result = handler(query=query_text, top_k=top_k)
        if not isinstance(result, list):
            result = [result]
        return result

    def register_handler(self, fn: "Callable") -> None:
        """
        Manually register a handler function for loopback mode.
        The @expose decorator calls this automatically.
        """
        self._registered_handler = fn


# ---------------------------------------------------------------------------
# ExposedHandler — the decorated callable
# ---------------------------------------------------------------------------


class ExposedHandler:
    """
    A handler wrapped by @provider.expose.

    Calling this object is identical to calling the original handler.
    It additionally exposes:
        .as_chorus_frame(*args, **kwargs)  → runs handler, returns CHORUSFrame
        .as_api_response(*args, **kwargs)  → runs handler, returns APIResponse
    """

    def __init__(self, fn: Callable, provider: PrismAPIProvider) -> None:
        self._fn = fn
        self._provider = provider
        functools.update_wrapper(self, fn)
        provider.register_handler(fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)

    def as_chorus_frame(self, *args: Any, **kwargs: Any) -> CHORUSFrame:
        """Run the handler and return the result as a CHORUSFrame."""
        result = self._fn(*args, **kwargs)
        if not isinstance(result, list):
            result = [result]
        return self._provider.as_chorus_frame(result)

    def as_api_response(self, *args: Any, **kwargs: Any) -> APIResponse:
        """Run the handler and return the result as an APIResponse."""
        result = self._fn(*args, **kwargs)
        if not isinstance(result, list):
            result = [result]
        return self._provider.project_results(result)


# ---------------------------------------------------------------------------
# ASGI adapter — plug a CHORUSframe endpoint into an existing FastAPI app
# ---------------------------------------------------------------------------


class ASGIAdapter:
    """
    Adds a ``/chorus/<handler_name>`` endpoint to an existing FastAPI app
    without touching any existing HTTP routes.

    Install::

        from prism.api.provider import ASGIAdapter
        adapter = ASGIAdapter(provider, handler_name="search")
        adapter.mount(app)     # app is a FastAPI instance

    The mounted endpoint accepts:
        POST /chorus/search
        Content-Type: application/x-chorus-frame
        Body: raw CHORUSFrame bytes (API_REQUEST)

    It responds with:
        Content-Type: application/x-chorus-frame
        Body: raw CHORUSFrame bytes (API_RESPONSE)

    The handler must accept a ``query_vector`` keyword argument OR a
    ``query`` string (if context["query_text"] is present).

    Architecture note:
        This does NOT replace the HTTP endpoint — it adds a parallel CHORUS
        channel.  Existing REST consumers continue to work unchanged.  This
        is the key adoption story: zero migration cost for existing clients.
    """

    CONTENT_TYPE = "application/x-chorus-frame"

    def __init__(
        self,
        handler: ExposedHandler,
        handler_name: str,
        route_prefix: str = "/chorus",
        auth: Optional[Any] = None,
        audit: Optional[Any] = None,
        rate_limiter: Optional[Any] = None,
    ) -> None:
        self._handler = handler
        self._name = handler_name
        self._prefix = route_prefix.rstrip("/")
        self._auth = auth
        self._audit = audit
        self._rate_limiter = rate_limiter

    def mount(self, app: Any) -> None:
        """
        Register the CHORUS endpoint on a FastAPI app.

        Requires: pip install fastapi
        """
        try:
            from fastapi import Request, Response  # type: ignore[import]
            from fastapi.routing import APIRoute  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "fastapi is required for ASGIAdapter.mount(): pip install fastapi"
            ) from exc

        path = f"{self._prefix}/{self._name}"
        handler = self._handler  # capture for closure

        async def chorus_endpoint(request: Request) -> Response:
            hdrs = dict(request.headers)
            from prism.api.auth import AuthConfig, actor_from_headers

            actor = actor_from_headers(hdrs)
            path = f"{self._prefix}/{self._name}"

            if self._rate_limiter is not None:
                from prism.security.rate_limit import RateLimitExceeded
                try:
                    self._rate_limiter.check(actor)
                except RateLimitExceeded:
                    if self._audit is not None:
                        self._audit.rate_limited(actor, resource=path)
                    return Response(content=b"Rate limit exceeded", status_code=429)

            if self._auth is not None:
                cfg = self._auth if isinstance(self._auth, AuthConfig) else self._auth
                if not cfg.validate_headers(hdrs):
                    if self._audit is not None:
                        self._audit.auth_denied(actor, resource=path, reason="invalid_credentials")
                    return Response(content=b"Unauthorized", status_code=401)
                if self._audit is not None:
                    self._audit.auth_success(actor, resource=path)

            body = await request.body()
            try:
                req_frame = CHORUSFrame.from_bytes(body)
                if req_frame.frame_type != FrameType.API_REQUEST:
                    return Response(
                        content=b"Expected API_REQUEST frame",
                        status_code=400,
                    )
                query_vec, ctx = req_frame.decode_api_request()
                query_text = ctx.get("query_text", "")
                top_k = int(ctx.get("top_k", 10))

                # Call the underlying handler
                if query_text:
                    result = handler(query=query_text, top_k=top_k)
                else:
                    result = handler(query_vector=query_vec, top_k=top_k)
                if not isinstance(result, list):
                    result = [result]

                resp_frame = handler._provider.as_chorus_frame(result)
                return Response(
                    content=resp_frame.to_bytes(),
                    media_type=self.CONTENT_TYPE,
                )
            except Exception as exc:
                logger.error("ASGIAdapter: error handling %s: %s", path, exc)
                return Response(content=str(exc).encode(), status_code=500)

        app.add_api_route(
            path,
            chorus_endpoint,
            methods=["POST"],
            response_class=type("RawResponse", (), {}),  # bypass FastAPI serialiser
        )
        logger.info("ASGIAdapter: mounted CHORUS endpoint at POST %s", path)
