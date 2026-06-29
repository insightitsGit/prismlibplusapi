"""Prometheus text exposition for Prism observability metrics."""

from __future__ import annotations

from prism.observability import get_registry


def prometheus_handler() -> tuple[bytes, str]:
    """Return (body, content_type) for a /metrics endpoint."""
    body = get_registry().to_prometheus().encode("utf-8")
    return body, "text/plain; version=0.0.4; charset=utf-8"
