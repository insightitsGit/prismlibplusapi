"""
prism.cache.cache — PrismCache: The Semantic LLM Cache
=======================================================

PrismCache is the main class developers interact with.
It wraps any LLM call and serves semantically similar queries
from the in-process wave cache instead of hitting the LLM again.

Quick start:
    from prism.cache import PrismCache, PrismCacheConfig
    from prism.cache.embedder import SentenceTransformerEmbedder

    cache = PrismCache.build(tenant_id="my-app")

    def call_llm(prompt: str) -> str:
        return openai_client.chat.completions.create(...).choices[0].message.content

    response = cache.get_or_call(
        query="What is your return policy?",
        call_fn=lambda: call_llm("What is your return policy?"),
    )

How it works:
    1. Embed the query text → float32 vector
    2. PrismLang projects it → 64-d tenant-isolated vector
    3. PrismResonance wave-interference query → find similar cached entries
    4. If similarity >= threshold: return cached response (microseconds)
    5. If no hit: call call_fn(), cache the result, return it

The tenant_id seeds the JL projection matrix via SHA-256(tenant_id).
Tenant A's cache entries are mathematically invisible to Tenant B's
queries — no query-time filter, no ACL, pure linear algebra isolation.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

import numpy as np

from prism.lib.lang import PrismProjector, ProjectionConfig
from prism.lib.resonance import PhaseState, PrismResonance, WavePacket
from prism.cache.embedder import Embedder, HashEmbedder, SentenceTransformerEmbedder
from prism.cache.metrics import (
    CacheMetrics,
    CostModel,
    MetricsCollector,
    get_cost_model,
)
from prism.cache.store import CacheEntry, CacheStore, InMemoryStore, SQLiteStore

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CacheError(Exception):
    """Base error for PrismCache operations."""


class CacheNotStartedError(CacheError):
    """Raised when cache methods are called before start() or within context."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PrismCacheConfig:
    """
    Configuration for a PrismCache instance.

    Attributes
    ----------
    tenant_id:
        Unique identifier for this tenant. Seeds the JL projection matrix
        via SHA-256(tenant_id). Two caches with different tenant_ids cannot
        see each other's entries — even if they share a store backend.

    similarity_threshold:
        Minimum wave-interference score [0, 1] to consider a cache hit.
        0.92 is a good default: tight enough to avoid wrong answers,
        loose enough to catch paraphrased queries.
        - 0.95+: very strict, fewer hits, highest answer accuracy
        - 0.90:  standard for most production use cases
        - 0.85:  aggressive caching, may serve slightly different questions
        from the same cache entry

    ttl_seconds:
        How long a cache entry lives before expiry.
        Default 3600 (1 hour). Set to 0 for no expiry.

    max_cache_size:
        Maximum number of WavePackets in the resonance store.
        When exceeded, the oldest packets are evicted.

    decay_interval_seconds:
        How often the background sleep cycle runs amplitude decay and eviction.

    llm_model:
        Name of the LLM model being cached. Used for cost estimation only.
        Does not affect cache behaviour.

    avg_tokens_per_response:
        Estimated token count per LLM response. Used for cost calculation.
        Override if you know your application's average.

    phase_state:
        PhaseState to assign new cache entries.
        ACTIVE for normal cached responses.
        ALERT for time-sensitive or high-priority entries.
        ARCHIVE for rarely-accessed historical entries.
    """

    tenant_id: str
    similarity_threshold: float = 0.92
    ttl_seconds: int = 3600
    max_cache_size: int = 50_000
    decay_interval_seconds: float = 300.0
    llm_model: str = "unknown"
    avg_tokens_per_response: int = 256
    phase_state: PhaseState = PhaseState.ACTIVE


# ---------------------------------------------------------------------------
# PrismCache
# ---------------------------------------------------------------------------


class PrismCache:
    """
    In-process semantic LLM cache with tenant isolation.

    Thread-safe: designed to be instantiated once and shared across all
    request-handling threads in your application.

    Lifecycle
    ---------
    Option 1 — context manager (recommended):
        async with PrismCache.build(tenant_id="acme") as cache:
            result = cache.get_or_call(query, call_fn)

    Option 2 — manual:
        cache = PrismCache.build(tenant_id="acme")
        await cache.start()
        ...
        await cache.stop()

    Option 3 — sync (no background decay cycle):
        cache = PrismCache.build(tenant_id="acme")
        result = cache.get_or_call(query, call_fn)
        # decay cycle does not run — entries expire by TTL only
    """

    def __init__(
        self,
        config: PrismCacheConfig,
        embedder: Embedder,
        store: CacheStore,
    ) -> None:
        self._cfg = config
        self._embedder = embedder
        self._store = store
        self._cost_model = get_cost_model(config.llm_model)
        self._metrics = MetricsCollector()

        # PrismLang projector — seeds tenant isolation from config.tenant_id
        self._projector = PrismProjector(
            ProjectionConfig(
                tenant_id=config.tenant_id,
                target_dim=64,
            )
        )

        # PrismResonance in-process wave cache
        self._resonance = PrismResonance(
            dim=64,
            decay_interval=config.decay_interval_seconds,
            extinction_threshold=0.001,  # very low — let TTL handle most expiry
        )

        self._started = False
        self._loop_thread: Optional[threading.Thread] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        tenant_id: str,
        *,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
        llm_model: str = "unknown",
        embedder: Optional[Embedder] = None,
        store: Optional[CacheStore] = None,
        persist_path: Optional[str] = None,
    ) -> "PrismCache":
        """
        Convenience factory — build a PrismCache with sensible defaults.

        Parameters
        ----------
        tenant_id:
            Your application or customer identifier.
        similarity_threshold:
            Minimum similarity for a cache hit [0, 1]. Default 0.92.
        ttl_seconds:
            Entry lifetime in seconds. Default 3600 (1 hour).
        llm_model:
            LLM model name for cost tracking. E.g. "gpt-4o", "claude-sonnet-4-6".
        embedder:
            Override the default embedder. Defaults to SentenceTransformerEmbedder
            (free, local) with HashEmbedder fallback if not installed.
        store:
            Override the default response store. Defaults to SQLiteStore if
            persist_path is provided, otherwise InMemoryStore.
        persist_path:
            If provided, use SQLiteStore at this path for response persistence.

        Usage:
            cache = PrismCache.build(
                tenant_id="acme",
                llm_model="gpt-4o",
                persist_path="/var/lib/prism/acme.db",
            )
        """
        cfg = PrismCacheConfig(
            tenant_id=tenant_id,
            similarity_threshold=similarity_threshold,
            ttl_seconds=ttl_seconds,
            llm_model=llm_model,
        )

        if embedder is None:
            try:
                embedder = SentenceTransformerEmbedder()
            except Exception:
                logger.warning(
                    "sentence-transformers not installed — using HashEmbedder "
                    "(not semantically meaningful). "
                    "Install with: pip install sentence-transformers"
                )
                embedder = HashEmbedder()

        if store is None:
            if persist_path:
                store = SQLiteStore(persist_path)
            else:
                store = InMemoryStore()

        return cls(cfg, embedder, store)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background decay/eviction loop."""
        await self._resonance.start()
        self._started = True
        logger.info(
            "PrismCache: started (tenant=%s, threshold=%.2f, ttl=%ds).",
            self._cfg.tenant_id,
            self._cfg.similarity_threshold,
            self._cfg.ttl_seconds,
        )

    async def stop(self) -> None:
        """Stop the background loop and flush pending metrics."""
        await self._resonance.stop()
        self._started = False
        logger.info("PrismCache: stopped.")

    async def __aenter__(self) -> "PrismCache":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Core API — synchronous
    # ------------------------------------------------------------------

    def get_or_call(
        self,
        query: str,
        call_fn: Callable[[], T],
        *,
        metadata: Optional[dict[str, Any]] = None,
        tokens_in_response: Optional[int] = None,
    ) -> T:
        """
        Return a cached response if a semantically similar query exists,
        otherwise call call_fn(), cache the result, and return it.

        Parameters
        ----------
        query:
            The user's query text. This is what gets embedded and matched.
        call_fn:
            Zero-argument callable that returns the LLM response.
            Only called on a cache miss.
        metadata:
            Optional dict attached to the cache entry for debugging.
        tokens_in_response:
            If you know the token count of the LLM response, pass it here
            for accurate cost tracking. Otherwise the config default is used.

        Returns
        -------
        The LLM response — either from cache or freshly generated.
        """
        from prism.observability.otel import trace_span

        with trace_span("prism.cache.get_or_call", tenant_id=self._cfg.tenant_id):
            t_start = time.monotonic()

            # Step 1: embed the query
            raw_embedding = self._embedder.embed(query)

            # Step 2: project into tenant-isolated 64-d space
            envelope = self._projector.project(raw_embedding)
            query_vector = envelope.vector  # float32, shape (64,)

            # Step 3: wave interference query — find similar entries
            query_packet = WavePacket.from_real_vector(
                query_vector,
                phase_state=self._cfg.phase_state,
            )
            hits = self._resonance.query(
                query_packet,
                top_k=1,
                amplitude_min=0.001,
            )

            # Step 4: check if best hit clears the threshold
            if hits and hits[0].constructive_score >= self._cfg.similarity_threshold:
                best_hit = hits[0]
                entry = self._store.load(best_hit.packet_id)

                if entry is not None:
                    # Cache hit
                    latency_ms = (time.monotonic() - t_start) * 1000
                    tokens = tokens_in_response or self._cfg.avg_tokens_per_response
                    cost_saved = self._cost_model.cost_for_tokens(tokens)

                    self._metrics.record(
                        was_hit=True,
                        latency_ms=latency_ms,
                        tokens_saved=tokens,
                        cost_saved_usd=cost_saved,
                        similarity_score=best_hit.constructive_score,
                        tenant_id=self._cfg.tenant_id,
                    )

                    logger.debug(
                        "PrismCache HIT  tenant=%s score=%.4f latency=%.2fms",
                        self._cfg.tenant_id,
                        best_hit.constructive_score,
                        latency_ms,
                    )
                    return entry.response  # type: ignore[return-value]

            # Step 5: cache miss — call the LLM
            t_llm_start = time.monotonic()
            response = call_fn()
            llm_latency_ms = (time.monotonic() - t_llm_start) * 1000

            # Step 6: store the response
            self._store_response(
                packet_id=envelope.envelope_id,
                query_vector=query_vector,
                query_text=query,
                response=response,
                metadata=metadata,
            )

            total_latency_ms = (time.monotonic() - t_start) * 1000
            self._metrics.record(
                was_hit=False,
                latency_ms=total_latency_ms,
                tokens_saved=0,
                cost_saved_usd=0.0,
                similarity_score=hits[0].constructive_score if hits else 0.0,
                tenant_id=self._cfg.tenant_id,
            )

            logger.debug(
                "PrismCache MISS tenant=%s llm_latency=%.0fms",
                self._cfg.tenant_id,
                llm_latency_ms,
            )
            return response  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Core API — async
    # ------------------------------------------------------------------

    async def aget_or_call(
        self,
        query: str,
        call_fn: Callable[[], Any],
        *,
        metadata: Optional[dict[str, Any]] = None,
        tokens_in_response: Optional[int] = None,
    ) -> Any:
        """
        Async version of get_or_call.

        call_fn may be a regular callable or a coroutine function.

        Usage:
            async def my_llm_call():
                response = await async_openai_client.chat.completions.create(...)
                return response.choices[0].message.content

            result = await cache.aget_or_call(
                query="What is the weather today?",
                call_fn=my_llm_call,
            )
        """
        from prism.observability.otel import trace_span

        with trace_span("prism.cache.aget_or_call", tenant_id=self._cfg.tenant_id):
            t_start = time.monotonic()

            raw_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self._embedder.embed, query
            )
            envelope = self._projector.project(raw_embedding)
            query_vector = envelope.vector

            query_packet = WavePacket.from_real_vector(
                query_vector, phase_state=self._cfg.phase_state
            )
            hits = self._resonance.query(query_packet, top_k=1, amplitude_min=0.001)

            if hits and hits[0].constructive_score >= self._cfg.similarity_threshold:
                entry = self._store.load(hits[0].packet_id)
                if entry is not None:
                    latency_ms = (time.monotonic() - t_start) * 1000
                    tokens = tokens_in_response or self._cfg.avg_tokens_per_response
                    self._metrics.record(
                        was_hit=True,
                        latency_ms=latency_ms,
                        tokens_saved=tokens,
                        cost_saved_usd=self._cost_model.cost_for_tokens(tokens),
                        similarity_score=hits[0].constructive_score,
                        tenant_id=self._cfg.tenant_id,
                    )
                    return entry.response

            # Miss — call the (possibly async) LLM function
            if asyncio.iscoroutinefunction(call_fn):
                response = await call_fn()
            else:
                response = await asyncio.get_event_loop().run_in_executor(None, call_fn)

            self._store_response(
                packet_id=envelope.envelope_id,
                query_vector=query_vector,
                query_text=query,
                response=response,
                metadata=metadata,
            )

            self._metrics.record(
                was_hit=False,
                latency_ms=(time.monotonic() - t_start) * 1000,
                tokens_saved=0,
                cost_saved_usd=0.0,
                similarity_score=hits[0].constructive_score if hits else 0.0,
                tenant_id=self._cfg.tenant_id,
            )
            return response

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_all(self) -> int:
        """
        Evict all entries for this tenant from the wave cache and store.

        Returns the number of wave packets evicted.
        Thread-safe.
        """
        before = self._resonance.count()
        # Snapshot IDs under lock, then delete one by one
        with self._resonance._lock:
            all_ids = list(self._resonance._store.keys())
        for pid in all_ids:
            try:
                self._resonance.delete(pid)
                self._store.delete(pid)
            except Exception:
                pass
        evicted = before - self._resonance.count()
        logger.info(
            "PrismCache.invalidate_all: evicted %d entries (tenant=%s).",
            evicted,
            self._cfg.tenant_id,
        )
        return evicted

    def purge_expired(self) -> int:
        """Remove expired entries from the response store. Returns count."""
        return self._store.purge_expired()

    def invalidate_by_doc_ids(self, doc_ids: set[str] | list[str]) -> int:
        """
        Evict cache entries linked to any of the given document IDs.

        Matches WavePacket metadata (doc_id, id) and query_text substrings.
        Returns the number of entries evicted.
        """
        targets = {str(d) for d in doc_ids if d}
        if not targets:
            return 0
        evicted = 0
        with self._resonance._lock:
            items = list(self._resonance._store.items())
        for pid, packet in items:
            meta = packet.metadata or {}
            refs = {str(meta.get("doc_id", "")), str(meta.get("id", ""))}
            entry = self._store.load(pid)
            text = entry.query_text if entry else str(meta.get("query_text", ""))
            if not targets.intersection(refs) and not any(t in text for t in targets):
                continue
            try:
                self._resonance.delete(pid)
                self._store.delete(pid)
                evicted += 1
            except Exception:
                pass
        if evicted:
            logger.info(
                "PrismCache.invalidate_by_doc_ids: evicted %d (tenant=%s).",
                evicted,
                self._cfg.tenant_id,
            )
        return evicted

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> CacheMetrics:
        """
        Return a point-in-time metrics snapshot.

        Includes hit rate, latency comparison, tokens saved, cost saved,
        and projected monthly savings. Thread-safe.
        """
        return self._metrics.snapshot()

    def print_metrics(self) -> None:
        """Print a formatted metrics summary to stdout."""
        print(self._metrics.snapshot().summary())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _store_response(
        self,
        packet_id: str,
        query_vector: np.ndarray,
        query_text: str,
        response: Any,
        metadata: Optional[dict[str, Any]],
    ) -> None:
        """
        Insert a new WavePacket into the resonance store and persist
        the response in the CacheStore backend.
        """
        now = time.time()
        expires_at = now + self._cfg.ttl_seconds if self._cfg.ttl_seconds > 0 else float("inf")

        # Insert into wave cache
        packet = WavePacket.from_real_vector(
            query_vector,
            phase_state=self._cfg.phase_state,
            amplitude=1.0,
            metadata={
                "query_text": query_text[:200],  # truncate for memory efficiency
                "tenant_id": self._cfg.tenant_id,
                **(metadata or {}),
            },
        )
        # Override packet_id so it matches the envelope_id (for store lookup)
        packet.packet_id = packet_id

        try:
            self._resonance.insert(packet)
        except Exception as exc:
            logger.warning("PrismCache: wave cache insert failed: %s", exc)
            return

        # Persist response to store
        entry = CacheEntry(
            packet_id=packet_id,
            query_text=query_text,
            response=response,
            created_at=now,
            expires_at=expires_at,
            model=self._cfg.llm_model,
        )
        try:
            self._store.save(entry)
        except Exception as exc:
            logger.warning("PrismCache: store save failed: %s", exc)

    @property
    def tenant_id(self) -> str:
        return self._cfg.tenant_id

    @property
    def cache_size(self) -> int:
        """Number of entries currently in the wave cache."""
        return self._resonance.count()
