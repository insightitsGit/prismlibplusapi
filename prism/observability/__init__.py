"""
prism.observability — Enterprise metrics and health registry.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MetricSnapshot:
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    help_text: str = ""


class MetricsRegistry:
    """
    Thread-safe in-process metrics registry.

    Exposes counters and gauges for Prometheus export and health endpoints.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._help: dict[str, str] = {}

    def counter(self, name: str, value: float = 1.0, *, help_text: str = "") -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value
            if help_text:
                self._help[name] = help_text

    def gauge(self, name: str, value: float, *, help_text: str = "") -> None:
        with self._lock:
            self._gauges[name] = value
            if help_text:
                self._help[name] = help_text

    def snapshot(self) -> list[MetricSnapshot]:
        with self._lock:
            out: list[MetricSnapshot] = []
            for k, v in self._counters.items():
                out.append(MetricSnapshot(k, v, help_text=self._help.get(k, "")))
            for k, v in self._gauges.items():
                out.append(MetricSnapshot(k, v, help_text=self._help.get(k, "")))
            return out

    def to_prometheus(self) -> str:
        lines: list[str] = []
        for m in self.snapshot():
            if m.help_text:
                lines.append(f"# HELP {m.name} {m.help_text}")
            lines.append(f"# TYPE {m.name} gauge")
            label_str = ""
            if m.labels:
                parts = [f'{k}="{v}"' for k, v in m.labels.items()]
                label_str = "{" + ",".join(parts) + "}"
            lines.append(f"{m.name}{label_str} {m.value}")
        return "\n".join(lines) + "\n"


_GLOBAL = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    return _GLOBAL


def record_cache_hit(*, tenant_id: str = "", latency_ms: float = 0.0) -> None:
    _GLOBAL.counter("prism_cache_hits_total", help_text="PrismCache semantic hits")
    _GLOBAL.gauge("prism_cache_last_hit_latency_ms", latency_ms)


def record_cache_miss(*, tenant_id: str = "", latency_ms: float = 0.0) -> None:
    _GLOBAL.counter("prism_cache_misses_total", help_text="PrismCache semantic misses")
    _GLOBAL.gauge("prism_cache_last_miss_latency_ms", latency_ms)


def record_driver_index(*, size: int, rows_received: int, rows_deleted: int) -> None:
    _GLOBAL.gauge("prism_driver_index_size", float(size))
    _GLOBAL.gauge("prism_driver_rows_received_total", float(rows_received))
    _GLOBAL.gauge("prism_driver_rows_deleted_total", float(rows_deleted))


def record_prismapi_request(*, provider: str = "", latency_ms: float = 0.0) -> None:
    _GLOBAL.counter("prism_api_requests_total", help_text="PrismAPI consumer requests")
    _GLOBAL.gauge("prism_api_last_request_latency_ms", latency_ms)


def health_payload(
    *,
    cache_metrics: Optional[dict[str, Any]] = None,
    driver_status: Optional[dict[str, Any]] = None,
    hub_status: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "ts": time.time(),
        "cache": cache_metrics or {},
        "driver": driver_status or {},
        "subscribe_hub": hub_status or {},
        "metrics": [m.__dict__ for m in _GLOBAL.snapshot()],
    }
