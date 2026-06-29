"""
benchmark.wrapper_sim.main — DB Node Simulator

Simulates the prism-wrapper Server Wrapper running on a DB node.
Holds an in-memory catalog of vectorized "product" rows and exposes:

  POST /wal/generate?count=N   — generate N synthetic rows into catalog
  GET  /wal/subscribe          — stream all catalog rows as WAL events (JSON)
  POST /query                  — direct vector similarity query (baseline path)
  GET  /catalog/stats          — catalog size and metadata
  GET  /health

This represents what the real prism-wrapper + DB would look like from the
app node's perspective: a remote service that owns the data and must be
queried over the network for every read.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TENANT_ID  = os.getenv("PRISM_TENANT_ID", "benchmark-db")
TARGET_DIM = int(os.getenv("PRISM_TARGET_DIM", "64"))

# ---------------------------------------------------------------------------
# Local catalog — simulates a vectorized DB table
# ---------------------------------------------------------------------------

class RowCatalog:
    """In-memory store of vectorized rows, mimicking what prism-wrapper builds
    from WAL events."""

    def __init__(self, tenant_id: str, dim: int = 64) -> None:
        self._tenant_id = tenant_id
        self._dim = dim
        self._rows: list[dict] = []          # {row_id, text_repr, vector}
        self._vectors: Optional[np.ndarray] = None  # (N, dim) matrix for batch ops
        self._dirty = True
        self._created_at = time.time()

    def _jl_seed(self) -> int:
        return int(hashlib.sha256(self._tenant_id.encode()).hexdigest()[:8], 16)

    def _project(self, raw: np.ndarray) -> np.ndarray:
        """Johnson-Lindenstrauss projection to TARGET_DIM."""
        rng = np.random.default_rng(self._jl_seed())
        if raw.size == 0 or np.linalg.norm(raw) == 0:
            raw = np.ones(1, dtype=np.float32)
        proj_matrix = rng.standard_normal((raw.size, self._dim)).astype(np.float32)
        proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True) + 1e-8
        v = (raw @ proj_matrix).astype(np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def generate(self, count: int) -> int:
        """Generate `count` synthetic product rows."""
        categories = [
            "electronics", "clothing", "furniture", "sports", "books",
            "kitchen", "toys", "automotive", "health", "garden",
        ]
        adjectives = [
            "premium", "deluxe", "compact", "ultra", "smart",
            "classic", "pro", "lite", "advanced", "eco",
        ]

        rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
        for i in range(count):
            cat  = categories[rng.integers(len(categories))]
            adj  = adjectives[rng.integers(len(adjectives))]
            name = f"{adj} {cat} item {len(self._rows) + 1}"
            price = float(rng.uniform(5, 500))
            stock = int(rng.integers(0, 200))

            # Build a raw numeric feature vector from the row fields
            cat_idx  = categories.index(cat) / len(categories)
            adj_idx  = adjectives.index(adj) / len(adjectives)
            raw = np.array([cat_idx, adj_idx, price / 500.0, stock / 200.0],
                           dtype=np.float32)
            vector = self._project(raw)

            self._rows.append({
                "row_id":    str(uuid.uuid4()),
                "text_repr": name,
                "price":     price,
                "stock":     stock,
                "category":  cat,
                "vector":    vector.tolist(),
            })

        self._dirty = True
        return count

    def _rebuild_matrix(self) -> None:
        if not self._rows:
            self._vectors = None
            return
        self._vectors = np.stack([np.array(r["vector"], dtype=np.float32)
                                  for r in self._rows])
        self._dirty = False

    def query(self, query_vector: list[float], top_k: int = 5,
              threshold: float = 0.7) -> list[dict]:
        """Cosine similarity search — this is what every baseline call does."""
        if self._dirty:
            self._rebuild_matrix()
        if self._vectors is None or len(self._rows) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-8
        normed = self._vectors / norms
        scores = normed @ q  # (N,)

        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < threshold:
                break
            row = self._rows[idx]
            results.append({
                "row_id":    row["row_id"],
                "text_repr": row["text_repr"],
                "score":     score,
                "category":  row["category"],
                "price":     row["price"],
            })
        return results

    def all_rows_for_streaming(self):
        """Yield all rows as WAL events for the subscribe endpoint."""
        for row in self._rows:
            yield row

    @property
    def size(self) -> int:
        return len(self._rows)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_catalog: RowCatalog = RowCatalog(TENANT_ID, TARGET_DIM)
_stats = {"total_direct_queries": 0, "total_subscribe_pulls": 0,
          "rows_streamed": 0, "started_at": time.time()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-generate 1000 rows at startup
    _catalog.generate(1000)
    logger.info("wrapper-sim: catalog pre-populated with %d rows", _catalog.size)
    yield
    logger.info("wrapper-sim: shutting down")


app = FastAPI(title="prism-wrapper-sim", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "catalog_size": _catalog.size,
            "tenant_id": TENANT_ID}


@app.post("/wal/generate")
def wal_generate(count: int = Query(default=1000, ge=1, le=50000)):
    """Generate additional rows into the catalog (simulates DB inserts)."""
    n = _catalog.generate(count)
    return {"generated": n, "catalog_size": _catalog.size}


@app.get("/wal/subscribe")
def wal_subscribe(limit: int = Query(default=5000, ge=1, le=100000)):
    """
    Stream catalog rows as newline-delimited JSON.
    The DLL Driver calls this to warm its local PrismResonance index.
    In the real system this is a persistent gRPC server-streaming RPC.
    """
    _stats["total_subscribe_pulls"] += 1

    def generate():
        rows = list(_catalog.all_rows_for_streaming())[:limit]
        _stats["rows_streamed"] += len(rows)
        for row in rows:
            # Emit each row as a WAL event JSON line
            yield json.dumps({
                "event_id":  str(uuid.uuid4()),
                "op":        "INSERT",
                "row_id":    row["row_id"],
                "text_repr": row["text_repr"],
                "vector":    row["vector"],
            }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/query")
def direct_query(body: dict):
    """
    Direct vector similarity query — the BASELINE path.
    Every call from the app node crosses the network to reach this endpoint.
    Simulates: app → network → DB node query → network → app.
    """
    t0 = time.perf_counter()
    query_vector = body.get("vector", [])
    top_k        = int(body.get("top_k", 5))
    threshold    = float(body.get("threshold", 0.5))

    results = _catalog.query(query_vector, top_k=top_k, threshold=threshold)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _stats["total_direct_queries"] += 1

    return {
        "results":    results,
        "elapsed_ms": elapsed_ms,
        "source":     "db-node-direct",
        "catalog_size": _catalog.size,
    }


@app.get("/catalog/stats")
def catalog_stats():
    return {**_stats, "catalog_size": _catalog.size,
            "uptime_s": time.time() - _stats["started_at"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
