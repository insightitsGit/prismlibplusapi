"""Benchmark app configuration — reads from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)


@dataclass
class BenchmarkConfig:
    # LLM
    openai_api_key:   str  = os.getenv("OPENAI_API_KEY", "")
    llm_model:        str  = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # PrismCache
    tenant_id:        str   = os.getenv("PRISM_TENANT_ID", "benchmark")
    similarity_threshold: float = float(os.getenv("PRISM_THRESHOLD", "0.92"))
    ttl_seconds:      int   = int(os.getenv("PRISM_TTL", "3600"))

    # Azure Application Insights
    appinsights_connection_string: str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")

    # Server
    host:  str = os.getenv("APP_HOST", "0.0.0.0")
    port:  int = int(os.getenv("APP_PORT", "8000"))

    # Benchmark mode — use mock LLM (no API key needed)
    use_mock_llm: bool = os.getenv("PRISM_BENCHMARK_SMOKE", "0") == "1" \
                      or not os.getenv("OPENAI_API_KEY", "")


_instance: BenchmarkConfig | None = None


def get_config() -> BenchmarkConfig:
    global _instance
    if _instance is None:
        _instance = BenchmarkConfig()
    return _instance
