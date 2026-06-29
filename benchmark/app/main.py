"""
Benchmark FastAPI app — wraps PrismCache around a real or mock LLM.

Endpoints:
  POST /query          — semantic cache lookup + LLM fallback
  POST /query/batch    — batch queries
  GET  /metrics        — live cache metrics snapshot
  GET  /health         — liveness probe
  POST /admin/seed     — load seed data into the cache
  POST /admin/reset    — flush the cache (new test run)
  GET  /admin/status   — cache size, mode, config

All requests are traced via Azure Application Insights (when configured).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from prism.cache import PrismCache, HashEmbedder
from benchmark.app.config import get_config
from benchmark.app.telemetry import setup_telemetry, trace_request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local index — the "PrismDriver" in-process vector store
# ---------------------------------------------------------------------------

WRAPPER_URL = os.getenv("PRISM_WRAPPER_URL", "")
TARGET_DIM  = int(os.getenv("PRISM_TARGET_DIM", "64"))
TENANT_ID_D = os.getenv("PRISM_TENANT_ID", "benchmark-app")

# Stats for baseline (network) path — driver path stats live on PrismDriver.local_index
_baseline_stats = {
    "queries": 0,
    "total_ms": 0.0,
}

# PrismDriver singleton — created in lifespan if WRAPPER_URL is set
_driver: Optional["Any"] = None


def get_driver():
    return _driver

# ---------------------------------------------------------------------------
# Cache singleton
# ---------------------------------------------------------------------------

_cache: PrismCache | None = None


def get_cache() -> PrismCache:
    if _cache is None:
        raise RuntimeError("Cache not initialised — app not started yet.")
    return _cache


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def make_llm_fn(question: str, cfg: Any):
    """Return a zero-arg callable that calls the LLM (or mock)."""
    if cfg.use_mock_llm:
        def mock_llm():
            # Deterministic mock: simulates 50-150ms LLM latency
            time.sleep(0.08)
            return f"[mock] Answer to: {question[:60]}"
        return mock_llm
    else:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.openai_api_key)
        def real_llm():
            resp = client.chat.completions.create(
                model=cfg.llm_model,
                messages=[{"role": "user", "content": question}],
                max_tokens=256,
            )
            return resp.choices[0].message.content
        return real_llm


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cache, _driver
    cfg = get_config()
    setup_telemetry(cfg.appinsights_connection_string)

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        embedder = None
    except ImportError:
        embedder = HashEmbedder(output_dim=384)
        logger.warning("sentence-transformers not installed — using HashEmbedder")

    _cache = PrismCache.build(
        tenant_id=cfg.tenant_id,
        llm_model=cfg.llm_model,
        similarity_threshold=cfg.similarity_threshold,
        ttl_seconds=cfg.ttl_seconds,
        embedder=embedder,
    )
    logger.info("PrismCache ready (tenant=%s, mock=%s)", cfg.tenant_id, cfg.use_mock_llm)

    # Start PrismDriver if a wrapper URL is configured.
    # The subscription loop begins immediately — by the time the first /driver/query
    # arrives, the local index will already be warm with WAL rows from the DB node.
    if WRAPPER_URL:
        from prism.ffi.bindings import PrismDriver, DriverConfig
        host = WRAPPER_URL.rstrip("/").split("//")[-1].split(":")[0]
        _driver = PrismDriver(DriverConfig(
            wrapper_host=host,
            wrapper_port=int(os.getenv("PRISM_WRAPPER_PORT", "8001")),
            tenant_id=TENANT_ID_D,
        ))
        await _driver.connect()
        logger.info(
            "PrismDriver started — subscription loop running, streaming from %s",
            WRAPPER_URL,
        )
    else:
        logger.info("PRISM_WRAPPER_URL not set — PrismDriver disabled")

    yield

    _cache.invalidate_all()
    if _driver is not None:
        await _driver.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="PrismLib Benchmark",
    description="Load-test harness for PrismCache semantic LLM cache",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    expected_tokens: int = Field(default=256, ge=1)

class QueryResponse(BaseModel):
    answer:     str
    cache_hit:  bool
    latency_ms: float
    similarity: float | None = None

class BatchQueryRequest(BaseModel):
    questions: list[str] = Field(..., min_items=1, max_items=100)

class BatchQueryResponse(BaseModel):
    results:        list[QueryResponse]
    total_ms:       float
    hits:           int
    misses:         int
    hit_rate_pct:   float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
@trace_request("query")
async def query(req: QueryRequest):
    cfg    = get_config()
    cache  = get_cache()
    t0     = time.monotonic()

    hits_before = cache.get_metrics().total_hits

    answer = await cache.aget_or_call(
        query=req.question,
        call_fn=make_llm_fn(req.question, cfg),
        tokens_in_response=req.expected_tokens,
    )

    latency_ms = (time.monotonic() - t0) * 1000
    was_hit    = cache.get_metrics().total_hits > hits_before

    return QueryResponse(
        answer=str(answer)[:500],
        cache_hit=was_hit,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/query/batch", response_model=BatchQueryResponse)
@trace_request("batch_query")
async def batch_query(req: BatchQueryRequest):
    cfg   = get_config()
    cache = get_cache()
    t0    = time.monotonic()

    tasks = [
        cache.aget_or_call(q, make_llm_fn(q, cfg))
        for q in req.questions
    ]
    answers = await asyncio.gather(*tasks)

    total_ms = (time.monotonic() - t0) * 1000
    m = cache.get_metrics()

    results = [
        QueryResponse(answer=str(a)[:200], cache_hit=False, latency_ms=0)
        for a in answers
    ]
    return BatchQueryResponse(
        results=results,
        total_ms=round(total_ms, 2),
        hits=m.total_hits,
        misses=m.total_misses,
        hit_rate_pct=round(m.hit_rate_pct, 1),
    )


@app.get("/metrics")
async def metrics():
    m = get_cache().get_metrics()
    return {
        "total_queries":          m.total_queries,
        "total_hits":             m.total_hits,
        "total_misses":           m.total_misses,
        "hit_rate_pct":           round(m.hit_rate_pct, 2),
        "avg_hit_latency_ms":     round(m.avg_hit_latency_ms, 2),
        "avg_miss_latency_ms":    round(m.avg_miss_latency_ms, 2),
        "speedup_factor":         round(m.speedup_factor, 1),
        "total_tokens_saved":     m.total_tokens_saved,
        "total_cost_saved_usd":   round(m.total_cost_saved_usd, 4),
        "projected_monthly_usd":  round(m.projected_monthly_savings_usd, 2),
        "cache_size":             get_cache().cache_size,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": get_cache().cache_size}


@app.post("/admin/seed")
async def seed(count: int = 1000):
    """Pre-load N Q&A pairs so the cache is warm before load testing."""
    from benchmark.load.seed_data import get_seed_questions
    cfg   = get_config()
    cache = get_cache()

    questions = get_seed_questions(count)
    loaded = 0
    for q in questions:
        await cache.aget_or_call(q, make_llm_fn(q, cfg))
        loaded += 1

    return {"seeded": loaded, "cache_size": cache.cache_size}


@app.post("/admin/reset")
async def reset():
    evicted = get_cache().invalidate_all()
    return {"evicted": evicted}


@app.get("/admin/status")
async def status():
    cfg = get_config()
    return {
        "tenant_id":           cfg.tenant_id,
        "llm_model":           cfg.llm_model,
        "similarity_threshold": cfg.similarity_threshold,
        "use_mock_llm":        cfg.use_mock_llm,
        "cache_size":          get_cache().cache_size,
    }


# ---------------------------------------------------------------------------
# PrismDriver endpoints
# (baseline = every query hits the wrapper over the network;
#  driver   = every query hits the local in-process index)
# ---------------------------------------------------------------------------

def _make_query_vector(text: str) -> np.ndarray:
    """Deterministic float32 query vector from text (mirrors wrapper_sim projection)."""
    h = hashlib.sha256(text.encode()).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(TARGET_DIM).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


@app.get("/driver/status")
async def driver_status():
    """Live state of PrismDriver subscription loop and local index."""
    d = get_driver()
    if d is None:
        return {"enabled": False, "reason": "PRISM_WRAPPER_URL not set"}
    return {"enabled": True, "wrapper_url": WRAPPER_URL, **d.index_status}


@app.post("/driver/baseline")
async def driver_baseline(text: str = Query(..., min_length=1)):
    """
    BASELINE path — proxy every query to the DB node over the network.
    Simulates direct DB access before PrismDriver is installed.
    Every call pays the full Azure inter-container round-trip.
    """
    if not WRAPPER_URL:
        raise HTTPException(502, "PRISM_WRAPPER_URL not set")

    t0 = time.monotonic()
    vector = _make_query_vector(text).tolist()

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{WRAPPER_URL}/query",
            json={"vector": vector, "top_k": 5, "threshold": 0.5},
        )
        resp.raise_for_status()
        data = resp.json()

    elapsed_ms = (time.monotonic() - t0) * 1000
    _baseline_stats["queries"] += 1
    _baseline_stats["total_ms"] += elapsed_ms

    return {
        "results":    data.get("results", []),
        "elapsed_ms": round(elapsed_ms, 2),
        "source":     "db-node-network",
    }


@app.post("/driver/query")
async def driver_query(text: str = Query(..., min_length=1)):
    """
    DRIVER path — query the local PrismResonance index via PrismDriver.
    The subscription loop has already streamed WAL rows into the index.
    Sub-millisecond, zero network hops.
    """
    d = get_driver()
    if d is None:
        raise HTTPException(503, "PrismDriver not running — set PRISM_WRAPPER_URL")

    if not d.local_index.is_warm:
        raise HTTPException(503, detail={
            "error": "local index not yet warm",
            "rows_received": d.local_index.rows_received,
            "hint": "subscription loop is streaming — retry in a few seconds",
        })

    t0 = time.perf_counter()
    vector = _make_query_vector(text)
    results, elapsed_ms = d.local_index.query(vector, top_k=5, threshold=0.5)

    return {
        "results":    [
            {"row_id": r.row_id, "text_repr": r.text_repr, "score": r.score}
            for r in results
        ],
        "elapsed_ms": round(elapsed_ms, 3),
        "source":     "local-index",
        "index_size": d.local_index.size,
    }


@app.get("/driver/metrics")
async def driver_metrics():
    """Comparison metrics: baseline (network) vs driver (local index)."""
    d = get_driver()
    bs_q   = _baseline_stats["queries"]
    bs_avg = (_baseline_stats["total_ms"] / bs_q) if bs_q else 0.0

    drv_q   = d.local_index.query_count if d else 0
    drv_avg = d.local_index.avg_latency_ms if d else 0.0
    speedup = (bs_avg / drv_avg) if drv_avg > 0 else 0.0

    return {
        "baseline": {
            "queries":        bs_q,
            "avg_latency_ms": round(bs_avg, 2),
            "source":         "db-node-network",
        },
        "driver": {
            "queries":        drv_q,
            "avg_latency_ms": round(drv_avg, 3),
            "source":         "local-index",
            "index_size":     d.local_index.size if d else 0,
            "rows_received":  d.local_index.rows_received if d else 0,
        },
        "speedup_factor": round(speedup, 1),
        "wrapper_url":    WRAPPER_URL or "not-configured",
        "sub_status":     d.index_status if d else None,
    }


@app.post("/driver/reset")
async def driver_reset():
    """Reset local index and baseline stats — start a fresh test run."""
    d = get_driver()
    evicted = d.local_index.reset() if d else 0
    _baseline_stats["queries"] = 0
    _baseline_stats["total_ms"] = 0.0
    return {"evicted": evicted}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(
        "benchmark.app.main:app",
        host=cfg.host,
        port=cfg.port,
        workers=1,
        log_level="info",
    )
