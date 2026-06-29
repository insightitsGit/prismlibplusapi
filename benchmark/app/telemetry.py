"""
Azure Application Insights telemetry setup.

When APPLICATIONINSIGHTS_CONNECTION_STRING is set, all FastAPI requests,
cache metrics, and custom events are automatically shipped to Azure Monitor.
Falls back to structured stdout logging when not configured.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

_telemetry_ready = False


def setup_telemetry(connection_string: str) -> None:
    global _telemetry_ready
    if not connection_string:
        logger.info("Telemetry: no APPLICATIONINSIGHTS_CONNECTION_STRING — stdout only.")
        return

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(connection_string=connection_string)
        _telemetry_ready = True
        logger.info("Telemetry: Azure Application Insights configured.")
    except ImportError:
        logger.warning(
            "Telemetry: azure-monitor-opentelemetry not installed — stdout only. "
            "Install with: pip install azure-monitor-opentelemetry"
        )


def track_metric(name: str, value: float, properties: dict | None = None) -> None:
    """Emit a custom metric to Application Insights (or log to stdout)."""
    if _telemetry_ready:
        try:
            from opentelemetry import metrics as otel_metrics
            meter = otel_metrics.get_meter("prism.benchmark")
            gauge = meter.create_gauge(name)
            gauge.set(value, properties or {})
        except Exception as exc:
            logger.debug("track_metric failed: %s", exc)
    else:
        props = " ".join(f"{k}={v}" for k, v in (properties or {}).items())
        logger.info("METRIC %s=%.4f %s", name, value, props)


def track_event(name: str, properties: dict | None = None) -> None:
    """Emit a custom event to Application Insights."""
    if _telemetry_ready:
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer("prism.benchmark")
            with tracer.start_as_current_span(name) as span:
                for k, v in (properties or {}).items():
                    span.set_attribute(k, str(v))
        except Exception as exc:
            logger.debug("track_event failed: %s", exc)
    else:
        props = " ".join(f"{k}={v}" for k, v in (properties or {}).items())
        logger.info("EVENT %s %s", name, props)


def trace_request(operation: str):
    """Decorator that adds latency + cache hit metrics to any endpoint."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.monotonic()
            try:
                result = await fn(*args, **kwargs)
                latency_ms = (time.monotonic() - t0) * 1000
                track_metric(
                    f"prism.{operation}.latency_ms",
                    latency_ms,
                    {"operation": operation},
                )
                return result
            except Exception as exc:
                track_event(f"prism.{operation}.error", {"error": str(exc)[:200]})
                raise
        return wrapper
    return decorator
