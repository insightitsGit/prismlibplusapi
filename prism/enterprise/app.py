"""
prism.enterprise.app — Enterprise FastAPI template with auth, metrics, health.
"""

from __future__ import annotations

from typing import Any, Optional


def create_enterprise_app(
    *,
    provider: Any,
    handler: Any,
    handler_name: str = "search",
    auth: Optional[Any] = None,
    title: str = "PrismAPI Enterprise",
) -> Any:
    """
    Build a FastAPI app with:
      - POST /chorus/{handler_name}  (PrismAPI + auth + rate limit + audit)
      - GET  /health
      - GET  /metrics  (Prometheus)
      - GET  /audit/recent  (last 100 security events)
    """
    try:
        from fastapi import FastAPI, Response
    except ImportError as exc:
        raise ImportError("pip install fastapi") from exc

    from prism.api.auth import AuthConfig
    from prism.api.provider import ASGIAdapter
    from prism.observability import health_payload
    from prism.observability.prometheus import prometheus_handler
    from prism.security.audit import AuditLogger
    from prism.security.rate_limit import RateLimitConfig, RateLimiter

    if auth is None:
        auth = AuthConfig.from_env()

    audit = AuditLogger(service=title)
    limiter = RateLimiter(RateLimitConfig(
        requests_per_minute=auth.rate_limit_rpm,
        burst=auth.rate_limit_burst,
    ))

    app = FastAPI(title=title)
    app.state.prism_audit = audit
    ASGIAdapter(
        handler,
        handler_name=handler_name,
        auth=auth,
        audit=audit,
        rate_limiter=limiter,
    ).mount(app)

    @app.get("/health")
    def health() -> dict:
        return health_payload()

    @app.get("/metrics")
    def metrics() -> Response:
        body, ctype = prometheus_handler()
        return Response(content=body, media_type=ctype)

    @app.get("/audit/recent")
    def audit_recent() -> list:
        return [e.__dict__ for e in audit.recent(100)]

    _ = provider  # reserved for future provider-level hooks
    return app
