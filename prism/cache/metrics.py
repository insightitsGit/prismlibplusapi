"""
prism.cache.metrics — Hit Rate, Latency, and Cost Savings Dashboard
====================================================================

PrismCache tracks every query in real time. This module collects those
signals into a CacheMetrics snapshot that tells you exactly how much
money and time the cache is saving.

The cost model uses per-token pricing for common LLM models. These prices
are updated periodically — override them via CostModel if you have
negotiated rates or use a different model.

Usage:
    metrics = cache.get_metrics()
    print(f"Hit rate:     {metrics.hit_rate_pct:.1f}%")
    print(f"Saved:        ${metrics.estimated_cost_saved_usd:.2f} today")
    print(f"Avg hit time: {metrics.avg_hit_latency_ms:.2f}ms")
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Cost model — price per 1M output tokens (USD)
# ---------------------------------------------------------------------------


@dataclass
class CostModel:
    """
    Per-token pricing for a specific LLM model.

    Prices are per 1,000,000 OUTPUT tokens (USD).
    Input tokens are typically cheaper but we track output tokens as the
    primary signal because they dominate the cost of cached responses.
    """

    model_name: str
    output_price_per_1m: float   # USD per 1M output tokens
    avg_tokens_per_response: int = 256  # estimated if not measured

    def cost_for_tokens(self, tokens: int) -> float:
        return (tokens / 1_000_000) * self.output_price_per_1m


# Prices as of mid-2025 — update these as models change
KNOWN_MODELS: dict[str, CostModel] = {
    # OpenAI
    "gpt-4o":                  CostModel("gpt-4o",                  10.00, 512),
    "gpt-4o-mini":             CostModel("gpt-4o-mini",              0.60, 256),
    "gpt-4-turbo":             CostModel("gpt-4-turbo",             30.00, 512),
    "gpt-3.5-turbo":           CostModel("gpt-3.5-turbo",            1.50, 256),
    # Anthropic
    "claude-opus-4-8":         CostModel("claude-opus-4-8",         75.00, 512),
    "claude-sonnet-4-6":       CostModel("claude-sonnet-4-6",       15.00, 512),
    "claude-haiku-4-5":        CostModel("claude-haiku-4-5",         1.25, 256),
    # Google
    "gemini-1.5-pro":          CostModel("gemini-1.5-pro",          10.50, 512),
    "gemini-1.5-flash":        CostModel("gemini-1.5-flash",         0.30, 256),
    # Meta / Open source
    "llama-3-70b":             CostModel("llama-3-70b",              0.90, 256),
}

DEFAULT_COST_MODEL = CostModel("unknown", output_price_per_1m=10.0, avg_tokens_per_response=256)


def get_cost_model(model_name: str) -> CostModel:
    """Return the CostModel for a model name, with fuzzy prefix matching."""
    if model_name in KNOWN_MODELS:
        return KNOWN_MODELS[model_name]
    # Prefix match (handles version suffixes)
    for key, model in KNOWN_MODELS.items():
        if model_name.startswith(key) or key.startswith(model_name):
            return model
    return DEFAULT_COST_MODEL


# ---------------------------------------------------------------------------
# Per-query event (internal)
# ---------------------------------------------------------------------------


@dataclass
class _QueryEvent:
    timestamp: float
    was_hit: bool
    latency_ms: float
    tokens_saved: int
    cost_saved_usd: float
    similarity_score: float
    tenant_id: str


# ---------------------------------------------------------------------------
# MetricsCollector (internal thread-safe event store)
# ---------------------------------------------------------------------------


class MetricsCollector:
    """
    Thread-safe rolling metrics collector.

    Stores the last `window_size` query events in a ring buffer.
    All public methods are safe to call from multiple threads.
    """

    def __init__(self, window_size: int = 10_000) -> None:
        self._window = window_size
        self._events: list[_QueryEvent] = []
        self._lock = threading.RLock()
        self._total_queries = 0
        self._total_hits = 0
        self._total_tokens_saved = 0
        self._total_cost_saved = 0.0
        self._started_at = time.time()

    def record(
        self,
        was_hit: bool,
        latency_ms: float,
        tokens_saved: int,
        cost_saved_usd: float,
        similarity_score: float,
        tenant_id: str,
    ) -> None:
        event = _QueryEvent(
            timestamp=time.time(),
            was_hit=was_hit,
            latency_ms=latency_ms,
            tokens_saved=tokens_saved,
            cost_saved_usd=cost_saved_usd,
            similarity_score=similarity_score,
            tenant_id=tenant_id,
        )
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._window:
                self._events.pop(0)
            self._total_queries += 1
            if was_hit:
                self._total_hits += 1
                self._total_tokens_saved += tokens_saved
                self._total_cost_saved += cost_saved_usd

    def snapshot(self) -> "CacheMetrics":
        """Return a point-in-time metrics snapshot. Thread-safe."""
        with self._lock:
            events = list(self._events)
            total_q = self._total_queries
            total_h = self._total_hits
            total_tokens = self._total_tokens_saved
            total_cost = self._total_cost_saved

        hit_events = [e for e in events if e.was_hit]
        miss_events = [e for e in events if not e.was_hit]

        def avg_latency(evts: list[_QueryEvent]) -> float:
            if not evts:
                return 0.0
            return sum(e.latency_ms for e in evts) / len(evts)

        # Last-hour stats from rolling window
        one_hour_ago = time.time() - 3600
        recent = [e for e in events if e.timestamp >= one_hour_ago]
        recent_hits = sum(1 for e in recent if e.was_hit)
        recent_cost = sum(e.cost_saved_usd for e in recent if e.was_hit)

        return CacheMetrics(
            total_queries=total_q,
            total_hits=total_h,
            total_misses=total_q - total_h,
            total_tokens_saved=total_tokens,
            total_cost_saved_usd=round(total_cost, 6),
            avg_hit_latency_ms=round(avg_latency(hit_events), 3),
            avg_miss_latency_ms=round(avg_latency(miss_events), 3),
            queries_last_hour=len(recent),
            hits_last_hour=recent_hits,
            cost_saved_last_hour_usd=round(recent_cost, 6),
            uptime_seconds=time.time() - self._started_at,
        )


# ---------------------------------------------------------------------------
# CacheMetrics (public snapshot)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheMetrics:
    """
    Immutable point-in-time snapshot of cache performance.

    All dollar values are in USD.
    All latency values are in milliseconds.

    Attributes
    ----------
    total_queries:          All queries ever received (hits + misses).
    total_hits:             Queries answered from cache.
    total_misses:           Queries that went to the LLM.
    total_tokens_saved:     Estimated output tokens saved by cache hits.
    total_cost_saved_usd:   Estimated USD saved by not calling the LLM.
    avg_hit_latency_ms:     Average response time for cache hits.
    avg_miss_latency_ms:    Average response time for cache misses (LLM calls).
    queries_last_hour:      Query volume in the rolling last 60 minutes.
    hits_last_hour:         Cache hits in the rolling last 60 minutes.
    cost_saved_last_hour_usd: USD saved in the rolling last 60 minutes.
    uptime_seconds:         Time since the cache was initialised.
    """

    total_queries: int
    total_hits: int
    total_misses: int
    total_tokens_saved: int
    total_cost_saved_usd: float
    avg_hit_latency_ms: float
    avg_miss_latency_ms: float
    queries_last_hour: int
    hits_last_hour: int
    cost_saved_last_hour_usd: float
    uptime_seconds: float

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction [0, 1]."""
        if self.total_queries == 0:
            return 0.0
        return self.total_hits / self.total_queries

    @property
    def hit_rate_pct(self) -> float:
        """Cache hit rate as a percentage [0, 100]."""
        return self.hit_rate * 100

    @property
    def speedup_factor(self) -> float:
        """
        How many times faster cache hits are compared to LLM calls.
        Returns 1.0 if no comparison data is available.
        """
        if self.avg_hit_latency_ms < 1e-3 or self.avg_miss_latency_ms < 1e-3:
            return 1.0
        return self.avg_miss_latency_ms / self.avg_hit_latency_ms

    @property
    def projected_monthly_savings_usd(self) -> float:
        """
        Extrapolate hourly savings to a full 30-day month.
        Based on the rolling last-hour window.
        """
        return self.cost_saved_last_hour_usd * 24 * 30

    def summary(self) -> str:
        """Return a human-readable one-paragraph summary."""
        lines = [
            f"PrismCache Metrics",
            f"  Hit rate:         {self.hit_rate_pct:.1f}%  "
            f"({self.total_hits:,} hits / {self.total_queries:,} queries)",
            f"  Speedup:          {self.speedup_factor:.0f}×  "
            f"({self.avg_hit_latency_ms:.1f}ms hit vs "
            f"{self.avg_miss_latency_ms:.0f}ms LLM call)",
            f"  Tokens saved:     {self.total_tokens_saved:,}",
            f"  Cost saved total: ${self.total_cost_saved_usd:.4f}",
            f"  Cost saved/hour:  ${self.cost_saved_last_hour_usd:.4f}",
            f"  Proj. monthly:    ${self.projected_monthly_savings_usd:.2f}",
            f"  Uptime:           {self.uptime_seconds / 3600:.1f}h",
        ]
        return "\n".join(lines)
