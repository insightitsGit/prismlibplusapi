"""
prism.api.auth — API key and bearer authentication for PrismAPI endpoints.
"""

from __future__ import annotations

import hmac
import os
import secrets
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from prism.security.audit import AuditLogger
    from prism.security.rate_limit import RateLimiter


class AuthError(Exception):
    """Raised when authentication fails."""


@dataclass
class AuthConfig:
    """
    Authentication configuration for PrismAPI ASGI endpoints.

    api_keys:
        Valid API keys (plain strings).  Compared with timing-safe equality.
    bearer_tokens:
        Valid bearer tokens (without the 'Bearer ' prefix).
    header_api_key:
        Header name for API key auth (default X-API-Key).
    require_auth:
        If True, reject requests with no credentials.  If False, auth is
        optional when api_keys/bearer_tokens are configured.
    """

    api_keys: tuple[str, ...] = field(default_factory=tuple)
    bearer_tokens: tuple[str, ...] = field(default_factory=tuple)
    header_api_key: str = "X-API-Key"
    require_auth: bool = True
    rate_limit_rpm: int = 120
    rate_limit_burst: int = 30

    @classmethod
    def from_env(cls, *, prefix: str = "PRISM_API") -> "AuthConfig":
        keys = os.environ.get(f"{prefix}_KEYS", "")
        tokens = os.environ.get(f"{prefix}_BEARER_TOKENS", "")
        require = os.environ.get(f"{prefix}_REQUIRE_AUTH", "true").lower() in (
            "1", "true", "yes",
        )
        rpm = int(os.environ.get(f"{prefix}_RATE_LIMIT_RPM", "120"))
        burst = int(os.environ.get(f"{prefix}_RATE_LIMIT_BURST", "30"))
        return cls(
            api_keys=tuple(k.strip() for k in keys.split(",") if k.strip()),
            bearer_tokens=tuple(t.strip() for t in tokens.split(",") if t.strip()),
            require_auth=require,
            rate_limit_rpm=rpm,
            rate_limit_burst=burst,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_keys or self.bearer_tokens)

    def validate_headers(self, headers: dict[str, str]) -> bool:
        """Return True if headers contain valid credentials."""
        if not self.enabled:
            return not self.require_auth

        lowered = {k.lower(): v for k, v in headers.items()}

        api_key = lowered.get(self.header_api_key.lower())
        if api_key and any(_safe_eq(api_key, k) for k in self.api_keys):
            return True

        auth = lowered.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            if any(_safe_eq(token, t) for t in self.bearer_tokens):
                return True

        return False


def _safe_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def generate_api_key() -> str:
    """Generate a URL-safe API key."""
    return secrets.token_urlsafe(32)


def actor_from_headers(headers: dict[str, str], header_api_key: str = "X-API-Key") -> str:
    """Derive audit actor id from request headers (API key prefix or client IP)."""
    lowered = {k.lower(): v for k, v in headers.items()}
    key = lowered.get(header_api_key.lower(), "")
    if key:
        return f"apikey:{key[:8]}"
    auth = lowered.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return f"bearer:{auth[7:13]}"
    return lowered.get("x-forwarded-for", lowered.get("x-real-ip", "anonymous"))


def auth_dependency(config: AuthConfig) -> Callable:
    """
    FastAPI dependency factory for PrismAPI routes.

    Usage::

        from fastapi import Depends
        from prism.api.auth import AuthConfig, auth_dependency

        auth = AuthConfig.from_env()
        app.add_api_route("/chorus/search", handler, dependencies=[Depends(auth_dependency(auth))])
    """
    def _check(headers: dict[str, str]) -> None:
        if not config.validate_headers(headers):
            raise AuthError("Unauthorized")

    async def _fastapi_dep(request: object) -> None:
        hdrs = {k: v for k, v in getattr(request, "headers", {}).items()}
        _check(hdrs)

    return _fastapi_dep
