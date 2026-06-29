"""
prism.security.rate_limit — Token-bucket rate limiting per API key / client IP.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class RateLimitExceeded(Exception):
    """Raised when a client exceeds its configured request rate."""


@dataclass
class RateLimitConfig:
    """
    Token-bucket rate limit.

    requests_per_minute:
        Sustained rate cap per bucket key (API key or client IP).
    burst:
        Maximum tokens in the bucket (allows short bursts).
    """

    requests_per_minute: int = 120
    burst: int = 30

    @property
    def refill_per_second(self) -> float:
        return self.requests_per_minute / 60.0


@dataclass
class _Bucket:
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class RateLimiter:
    """Thread-safe per-key token bucket."""

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self._cfg = config or RateLimitConfig()
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.RLock()

    def check(self, key: str, *, cost: float = 1.0) -> None:
        """Consume tokens or raise RateLimitExceeded."""
        if not key:
            key = "anonymous"
        with self._lock:
            now = time.monotonic()
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self._cfg.burst))
                self._buckets[key] = bucket
            elapsed = now - bucket.last_refill
            bucket.tokens = min(
                float(self._cfg.burst),
                bucket.tokens + elapsed * self._cfg.refill_per_second,
            )
            bucket.last_refill = now
            if bucket.tokens < cost:
                raise RateLimitExceeded(
                    f"Rate limit exceeded for key={key[:8]}... "
                    f"({self._cfg.requests_per_minute}/min)"
                )
            bucket.tokens -= cost

    def reset(self, key: str) -> None:
        with self._lock:
            self._buckets.pop(key, None)
