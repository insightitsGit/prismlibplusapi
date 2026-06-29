"""
benchmark/api/bench_server.py
==============================

Standalone HTTP server exposing two endpoints for the real benchmark:

    GET  /health                        → 200 OK when ready
    GET  /search?q=<query>&top_k=<N>   → JSON (BASELINE path)
    POST /chorus/search                 → CHORUSFrame bytes (PRISMAPI path)

Run standalone (for debugging):
    python benchmark/api/bench_server.py --port 9101 --dim 64

The benchmark runner (run_real_benchmark.py) starts this as a subprocess,
polls /health, then fires HTTP requests from the client process.

Design notes
------------
- Uses only Python stdlib (http.server, json, urllib) — no FastAPI/uvicorn dep
- Loads sentence-transformers once at startup
- Returns real embedding latency in X-Embed-Ms response header
- Both endpoints share the same corpus embedding index (pre-built at startup)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from functools import lru_cache
import threading
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from prism.lib.lang import PrismProjector, ProjectionConfig
from prism.api import PrismAPIProvider, SentenceTransformerEmbedder
from benchmark.api.bench_corpus import CORPUS


# ---------------------------------------------------------------------------
# Global server state (set by main())
# ---------------------------------------------------------------------------

_embedder: SentenceTransformerEmbedder | None = None
_projector: PrismProjector | None = None
_provider: PrismAPIProvider | None = None
_corpus_embeddings: np.ndarray | None = None     # (N, embed_dim) full-dim, normalised
_corpus_projected: np.ndarray | None = None      # (N, target_dim) projected, normalised
_corpus_projected_raw: np.ndarray | None = None  # (N, target_dim) projected, un-normalised (for packing)
_target_dim: int = 64


def _cosine_search_fullrank(query_emb: np.ndarray, top_k: int) -> list[int]:
    """Baseline: cosine in full embedding space (384-dim). Returns corpus indices."""
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    scores = _corpus_embeddings @ q
    return list(np.argsort(scores)[::-1][:top_k])


def _cosine_search_projected(query_emb: np.ndarray, top_k: int) -> list[int]:
    """PrismAPI: cosine in projected space (target_dim). Returns corpus indices."""
    env = _projector.project(query_emb)
    q = env.vector.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    scores = _corpus_projected @ q
    return list(np.argsort(scores)[::-1][:top_k])


def _pack_chorus_response(indices: list[int]) -> bytes:
    """
    Build a CHORUSFrame API_RESPONSE directly from the pre-projected index.
    NO embedding call here — vectors were computed at startup and cached.
    This is the correct architecture: index once, serve many.
    """
    from prism.lib.fabric import CHORUSFrame

    results = []
    for i in indices:
        doc = CORPUS[i]
        vec = _corpus_projected_raw[i]   # pre-computed, no embed call
        sidecar = {"doc_id": doc["doc_id"], "domain": doc["domain"], "source": doc["source"]}
        results.append((vec, sidecar))

    frame = CHORUSFrame.from_api_response(
        key_id=_provider._cipher._active_key.key_id,
        seq=0,
        watermark=b"\x00" * 32,
        results=results,
    )
    return frame.to_bytes()


def _cosine_search_projected_vec(query_vec: np.ndarray, top_k: int) -> list[int]:
    """
    Search in projected space using a pre-projected query vector directly.
    No embedding call — the client already projected and sent the vector.
    This is the key optimisation: when the client sends a float32 vector
    in the CHORUSFrame, the server uses it as-is. Zero server embed calls.
    """
    q = query_vec.astype(np.float32)
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q = q / norm
    scores = _corpus_projected @ q
    return list(np.argsort(scores)[::-1][:top_k])


# Thread-safe LRU cache for baseline query embeddings.
# In concurrent load, many agents send identical or near-identical queries.
# Caching avoids redundant embedding calls under load.
_embed_cache: dict[str, np.ndarray] = {}
_embed_cache_lock = threading.Lock()
_EMBED_CACHE_MAX = 512


def _embed_query_cached(query: str) -> np.ndarray:
    """Return cached embedding or compute and cache it (thread-safe)."""
    with _embed_cache_lock:
        if query in _embed_cache:
            return _embed_cache[query]
    # Compute outside the lock — embedding is slow, don't block other threads
    emb = _embedder.embed([query])[0]
    with _embed_cache_lock:
        if len(_embed_cache) >= _EMBED_CACHE_MAX:
            # Evict oldest entry (FIFO approximation)
            oldest = next(iter(_embed_cache))
            del _embed_cache[oldest]
        _embed_cache[query] = emb
    return emb


# ---------------------------------------------------------------------------
# Threaded HTTP server — handles concurrent requests in separate threads
# ---------------------------------------------------------------------------

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """
    Spawns a thread per request.

    Why: http.server.HTTPServer is single-threaded — under concurrent load
    requests queue up and wait for the previous one to finish. ThreadingMixIn
    dispatches each connection to a new thread immediately. The GIL means
    CPU-bound numpy ops still contend, but I/O waits (embedding model load,
    socket reads) are released, so real concurrency is achieved.
    """
    daemon_threads = True   # threads exit when main process exits


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class BenchHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass   # suppress per-request access logs

    # ------------------------------------------------------------------
    # GET dispatcher
    # ------------------------------------------------------------------

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._send_json({"status": "ok", "corpus": len(CORPUS)}, 200)

        elif path == "/search":
            qs = urllib.parse.parse_qs(parsed.query)
            query = qs.get("q", [""])[0]
            top_k = int(qs.get("top_k", ["5"])[0])
            self._handle_baseline(query, top_k)

        else:
            self._send_json({"error": "not found"}, 404)

    # ------------------------------------------------------------------
    # POST dispatcher
    # ------------------------------------------------------------------

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/chorus/search":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            self._handle_chorus(body)
        else:
            self._send_json({"error": "not found"}, 404)

    # ------------------------------------------------------------------
    # Baseline: embed query → full-rank cosine → return JSON with text
    # ------------------------------------------------------------------

    def _handle_baseline(self, query: str, top_k: int) -> None:
        if not query:
            self._send_json({"error": "missing q"}, 400)
            return

        t_embed_start = time.perf_counter()
        query_emb = _embed_query_cached(query)
        t_embed_ms = (time.perf_counter() - t_embed_start) * 1000.0

        indices = _cosine_search_fullrank(query_emb, top_k)

        # Return full text fields — the client MUST embed these to use them
        payload = json.dumps({
            "query": query,
            "top_k": top_k,
            "results": [
                {
                    "doc_id": CORPUS[i]["doc_id"],
                    "title": CORPUS[i]["title"],
                    "body": CORPUS[i]["body"],
                    "domain": CORPUS[i]["domain"],
                    "source": CORPUS[i]["source"],
                }
                for i in indices
            ],
        }).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Embed-Ms", f"{t_embed_ms:.2f}")
        self.end_headers()
        self.wfile.write(payload)

    # ------------------------------------------------------------------
    # PrismAPI: decode CHORUSFrame request → projected search → CHORUSFrame response
    # ------------------------------------------------------------------

    def _handle_chorus(self, body: bytes) -> None:
        from prism.lib.fabric import CHORUSFrame, FrameType

        try:
            t_embed_start = time.perf_counter()
            req_frame = CHORUSFrame.from_bytes(body)

            if req_frame.frame_type != FrameType.API_REQUEST:
                self._send_json({"error": "expected API_REQUEST frame"}, 400)
                return

            query_vec, ctx = req_frame.decode_api_request()
            top_k = int(ctx.get("top_k", 5))
            t_embed_ms = 0.0   # server does NO embedding on the CHORUS path

            # KEY: use the client's pre-projected query vector directly.
            # The client embedded its query and projected it to target_dim before
            # sending. The server searches in the same projected space with zero
            # embedding calls. This is the data-type separation: CHORUS frames
            # carry float32 vectors, not text — the server never touches the
            # embedding model for this path.
            indices = _cosine_search_projected_vec(query_vec, top_k)

            # Pack pre-projected vectors from the index — NO re-embedding
            resp_bytes = _pack_chorus_response(indices)

        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-chorus-frame")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.send_header("X-Embed-Ms", f"{t_embed_ms:.2f}")
        self.end_headers()
        self.wfile.write(resp_bytes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_json(self, data: Any, code: int) -> None:
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def build_index(embedder: SentenceTransformerEmbedder, projector: PrismProjector) -> None:
    global _corpus_embeddings, _corpus_projected, _corpus_projected_raw

    print(f"[server] Building index for {len(CORPUS)} documents...", flush=True)
    texts = [f"{d['title']} | {d['body']}" for d in CORPUS]

    t0 = time.perf_counter()
    raw = embedder.embed(texts)   # (N, embed_dim) — one real batch embedding call
    t1 = time.perf_counter()
    print(f"[server] Embedded {len(CORPUS)} docs in {(t1-t0)*1000:.0f} ms", flush=True)

    # Normalise for cosine via dot product
    norms = np.linalg.norm(raw, axis=1, keepdims=True).clip(min=1e-9)
    _corpus_embeddings = (raw / norms).astype(np.float32)

    # Project to target_dim
    projected = np.stack([projector.project(raw[i]).vector for i in range(len(CORPUS))], axis=0)
    _corpus_projected_raw = projected.astype(np.float32)   # un-normalised, for CHORUSFrame packing
    p_norms = np.linalg.norm(projected, axis=1, keepdims=True).clip(min=1e-9)
    _corpus_projected = (projected / p_norms).astype(np.float32)   # normalised, for cosine search

    print(f"[server] Index ready. "
          f"Full-rank shape: {_corpus_embeddings.shape}, "
          f"Projected shape: {_corpus_projected.shape}", flush=True)


def main() -> None:
    global _embedder, _projector, _provider, _target_dim

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    _target_dim = args.dim

    print(f"[server] Loading embedder ({args.model})...", flush=True)
    _embedder = SentenceTransformerEmbedder(model_name=args.model)
    print(f"[server] Embed dim: {_embedder.embed_dim}", flush=True)

    _projector = PrismProjector(ProjectionConfig(
        tenant_id="bench-tenant",
        target_dim=args.dim,
    ))

    _provider = PrismAPIProvider(
        projector=_projector,
        embedder=_embedder,
        semantic_fields=["title", "body"],
        id_field="doc_id",
        exact_fields=["domain", "source"],
        provider_id="bench-server",
    )

    build_index(_embedder, _projector)

    server = ThreadedHTTPServer(("127.0.0.1", args.port), BenchHandler)
    print(f"[server] Listening on 127.0.0.1:{args.port} (target_dim={args.dim}, threaded)", flush=True)
    print("[server] READY", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
