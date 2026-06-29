"""
prism.observability.otel — Optional OpenTelemetry instrumentation.

Install: pip install "prismlib-plus[otel]"

Usage::

    from prism.observability.otel import configure_tracing, trace_span

    configure_tracing(service_name="my-app")
    with trace_span("prism.cache.get_or_call"):
        ...
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

_tracer: Any = None
_configured = False


def configure_tracing(
    service_name: str = "prismlib",
    *,
    otlp_endpoint: Optional[str] = None,
) -> bool:
    """
    Configure OpenTelemetry tracing if opentelemetry-sdk is installed.

    Returns True if tracing was configured, False if OTel is not available.
    """
    global _tracer, _configured
    if _configured:
        return _tracer is not None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.info("OpenTelemetry not installed — tracing disabled")
        _configured = True
        return False

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
        except ImportError:
            logger.warning("OTLP exporter not installed — using console spans only")

    try:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    except Exception:
        pass

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("prism")
    _configured = True
    logger.info("OpenTelemetry tracing configured for service=%s", service_name)
    return True


@contextmanager
def trace_span(name: str, **attributes: Any) -> Generator[None, None, None]:
    """Context manager that creates an OTel span when tracing is configured."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as span:
        for k, v in attributes.items():
            span.set_attribute(k, v)
        yield
