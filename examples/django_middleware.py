"""
Example: PrismCache as Django Middleware
=========================================

Drop this into your Django project to automatically cache all LLM
calls made during request handling.

Setup in settings.py:
    MIDDLEWARE = [
        ...
        'examples.django_middleware.PrismCacheMiddleware',
    ]

    PRISM_CACHE = {
        'TENANT_ID': 'my-django-app',
        'LLM_MODEL': 'gpt-4o',
        'SIMILARITY_THRESHOLD': 0.92,
        'TTL_SECONDS': 3600,
        'PERSIST_PATH': '/var/lib/prism/cache.db',
    }
"""

from __future__ import annotations

from typing import Any, Callable
from prism.cache import PrismCache

# Module-level singleton — shared across all request threads
_cache: PrismCache | None = None


def get_cache() -> PrismCache:
    """Return the module-level PrismCache singleton."""
    global _cache
    if _cache is None:
        try:
            from django.conf import settings  # type: ignore[import]
            cfg = getattr(settings, "PRISM_CACHE", {})
        except ImportError:
            cfg = {}

        _cache = PrismCache.build(
            tenant_id=cfg.get("TENANT_ID", "django-app"),
            llm_model=cfg.get("LLM_MODEL", "unknown"),
            similarity_threshold=cfg.get("SIMILARITY_THRESHOLD", 0.92),
            ttl_seconds=cfg.get("TTL_SECONDS", 3600),
            persist_path=cfg.get("PERSIST_PATH"),
        )
    return _cache


class PrismCacheMiddleware:
    """
    Django middleware that attaches the PrismCache to each request.

    After adding to MIDDLEWARE, access the cache in any view:
        def my_view(request):
            answer = request.prism_cache.get_or_call(
                query=user_question,
                call_fn=lambda: call_my_llm(user_question),
            )
    """

    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response
        self.cache = get_cache()

    def __call__(self, request: Any) -> Any:
        request.prism_cache = self.cache
        response = self.get_response(request)
        return response


# ── Standalone usage (no Django) ──────────────────────────────────────────────
if __name__ == "__main__":
    cache = get_cache()
    print("PrismCache ready for Django middleware.")
    print(f"  Tenant: {cache.tenant_id}")
    print(f"  Cache size: {cache.cache_size} entries")
