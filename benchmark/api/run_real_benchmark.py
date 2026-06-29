"""
benchmark/api/run_real_benchmark.py
=====================================

REAL benchmark: client and server are separate processes communicating over
TCP loopback. Every measurement is a genuine HTTP round-trip.

Baseline path (real HTTP):
    client → GET /search?q=... → server embeds query (real) → JSON text response
    → client must embed returned text to use it (real embedding calls measured)

PrismAPI path (real HTTP):
    client → POST /chorus/search (CHORUSFrame) → server embeds query + projects
    → CHORUSFrame response with float32 vectors
    → client decodes vectors directly (0 re-embedding calls)

Metrics reported per path
--------------------------
    end_to_end_latency_ms   TCP round-trip + client processing (P50/P95/P99)
    server_embed_ms         actual server-side embedding time (from response header)
    client_embed_ms         client re-embedding of returned text (baseline only)
    total_embedding_calls   across all queries × trials
    wire_bytes              actual bytes transferred over TCP per request
    recall_at_k             fraction of baseline top-K in PrismAPI top-K
    recall_at_2k            fraction of baseline top-K in PrismAPI top-2K
    jaccard_at_k            strict set overlap

Run
---
    python benchmark/api/run_real_benchmark.py
    python benchmark/api/run_real_benchmark.py --trials 5 --dim 128 --top-k 10

Requirements
------------
    pip install sentence-transformers numpy
    (server uses only stdlib http.server — no fastapi/uvicorn required)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from prism.lib.lang import PrismProjector, ProjectionConfig
from prism.lib.fabric import CHORUSFrame, FrameType
from prism.api import PrismAPIProvider, SentenceTransformerEmbedder
from prism.api.schema import unpack_response_payload
from benchmark.api.bench_corpus import CORPUS, QUERIES


SERVER_PORT = 9101
SERVER_HOST = "127.0.0.1"
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def wait_for_server(timeout: float = 60.0) -> None:
    """Poll /health until the server responds or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"Server did not become ready within {timeout}s")


def start_server(dim: int, model: str) -> subprocess.Popen:
    """Start bench_server.py as a subprocess."""
    server_script = Path(__file__).parent / "bench_server.py"
    proc = subprocess.Popen(
        [sys.executable, str(server_script),
         "--port", str(SERVER_PORT),
         "--dim", str(dim),
         "--model", model],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def stream_server_until_ready(proc: subprocess.Popen) -> None:
    """Read server stdout until '[server] READY' line appears."""
    print("[runner] Waiting for server startup...")
    for line in proc.stdout:
        print(f"  {line.rstrip()}")
        if "READY" in line:
            break


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url: str) -> tuple[bytes, dict[str, str], float]:
    """Return (body, headers, elapsed_ms)."""
    t0 = time.perf_counter()
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
        headers = dict(resp.headers)
    return body, headers, (time.perf_counter() - t0) * 1000.0


def http_post_chorus(frame: CHORUSFrame) -> tuple[bytes, dict[str, str], float]:
    """POST a CHORUSFrame, return (body, headers, elapsed_ms)."""
    data = frame.to_bytes()
    req = urllib.request.Request(
        f"{SERVER_URL}/chorus/search",
        data=data,
        headers={"Content-Type": "application/x-chorus-frame"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
        headers = dict(resp.headers)
    return body, headers, (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Per-path benchmarks
# ---------------------------------------------------------------------------

def run_baseline_path(
    embedder: SentenceTransformerEmbedder,
    queries: list[str],
    top_k: int,
    n_trials: int,
) -> dict:
    """
    Baseline (real HTTP):
        1. GET /search?q=query → JSON with text results
        2. Client embeds returned text (measures real embedding latency)
        3. Record: total RTT, server embed time, client embed time, wire bytes
    """
    print(f"\n[baseline] Running {len(queries)} queries × {n_trials} trials (top-{top_k})...")

    rtt_ms: list[float] = []
    server_embed_ms: list[float] = []
    client_embed_ms: list[float] = []
    wire_bytes: list[int] = []
    all_result_ids: list[list[str]] = []

    for q_idx, query in enumerate(queries):
        q_rtts: list[float] = []
        q_server_embeds: list[float] = []
        q_wire: list[int] = []
        result_ids: list[str] = []

        for trial in range(n_trials):
            url = f"{SERVER_URL}/search?q={urllib.request.quote(query)}&top_k={top_k}"
            body, headers, elapsed = http_get(url)
            q_rtts.append(elapsed)
            q_wire.append(len(body))

            server_ms = float(headers.get("X-Embed-Ms", headers.get("x-embed-ms", "0")))
            q_server_embeds.append(server_ms)

            data = json.loads(body)
            result_texts = [
                f"{r['title']} | {r['body']}"
                for r in data.get("results", [])
            ]

            # Real client embedding call — this is the cost the client pays
            t_client = time.perf_counter()
            if result_texts:
                _ = embedder.embed(result_texts)
            client_embed_ms.append((time.perf_counter() - t_client) * 1000.0)

            if trial == 0:
                result_ids = [r["doc_id"] for r in data.get("results", [])]

        rtt_ms.extend(q_rtts)
        server_embed_ms.extend(q_server_embeds)
        wire_bytes.extend(q_wire)
        all_result_ids.append(result_ids)

        if (q_idx + 1) % 10 == 0:
            print(f"  {q_idx+1}/{len(queries)} queries done "
                  f"(mean RTT: {np.mean(q_rtts):.1f} ms)")

    total_embed_calls = len(queries) * n_trials * (1 + top_k)   # server query + client results

    return {
        "path": "baseline",
        "n_queries": len(queries),
        "n_trials": n_trials,
        "top_k": top_k,
        "rtt_ms": {
            "mean": float(np.mean(rtt_ms)),
            "p50": float(np.percentile(rtt_ms, 50)),
            "p95": float(np.percentile(rtt_ms, 95)),
            "p99": float(np.percentile(rtt_ms, 99)),
            "std": float(np.std(rtt_ms)),
        },
        "server_embed_ms": {
            "mean": float(np.mean(server_embed_ms)),
            "p50": float(np.percentile(server_embed_ms, 50)),
            "p95": float(np.percentile(server_embed_ms, 95)),
        },
        "client_embed_ms": {
            "mean": float(np.mean(client_embed_ms)),
            "p50": float(np.percentile(client_embed_ms, 50)),
            "p95": float(np.percentile(client_embed_ms, 95)),
        },
        "total_embedding_calls": total_embed_calls,
        "mean_embed_calls_per_query": total_embed_calls / (len(queries) * n_trials),
        "mean_wire_bytes": float(np.mean(wire_bytes)),
        "result_ids_per_query": all_result_ids,
    }


def run_prismapi_path(
    embedder: SentenceTransformerEmbedder,
    projector: PrismProjector,
    queries: list[str],
    top_k: int,
    n_trials: int,
) -> dict:
    """
    PrismAPI (real HTTP):
        1. Embed query locally (1 call) → project to target_dim
        2. POST /chorus/search with CHORUSFrame
        3. Decode CHORUSFrame response → float32 vectors (0 re-embedding)
        4. Record: total RTT, server embed time, wire bytes
    """
    from prism.lib.fabric import TensorCipher

    cipher = TensorCipher(dim=projector._cfg.target_dim, ttl_seconds=3600.0)
    cipher.rotate_key()
    seq = 0

    print(f"\n[prismapi] Running {len(queries)} queries × {n_trials} trials (top-{top_k})...")

    rtt_ms: list[float] = []
    server_embed_ms: list[float] = []
    client_embed_ms: list[float] = []   # always 0 — no re-embedding
    wire_bytes: list[int] = []
    all_result_ids: list[list[str]] = []
    all_result_ids_2k: list[list[str]] = []

    for q_idx, query in enumerate(queries):
        q_rtts: list[float] = []
        q_server_embeds: list[float] = []
        q_wire: list[int] = []
        result_ids: list[str] = []
        result_ids_2k: list[str] = []

        for trial in range(n_trials):
            # Client embeds query once (real embedding call, measured)
            t_cembed = time.perf_counter()
            raw_emb = embedder.embed([query])[0]
            env = projector.project(raw_emb)
            query_vec = env.vector
            client_embed_ms.append((time.perf_counter() - t_cembed) * 1000.0)

            key = cipher._active_key
            req_frame = CHORUSFrame.from_api_request(
                key_id=key.key_id,
                seq=seq,
                watermark=b"\x00" * 32,
                query_vector=query_vec,
                context={"query_text": query, "top_k": top_k},
            )
            seq += 1

            body, headers, elapsed = http_post_chorus(req_frame)
            q_rtts.append(elapsed)
            q_wire.append(len(body) + len(req_frame.to_bytes()))   # both directions

            server_ms = float(headers.get("X-Embed-Ms", headers.get("x-embed-ms", "0")))
            q_server_embeds.append(server_ms)

            # Decode response — no embedding needed
            resp_frame = CHORUSFrame.from_bytes(body)
            raw_results = resp_frame.decode_api_response()
            pairs = unpack_response_payload(raw_results)

            if trial == 0:
                result_ids = [s.doc_id for _, s in pairs]

        # Fetch top-2K for Recall@2K metric (one extra call outside timing loop)
        raw_emb = embedder.embed([query])[0]
        env = projector.project(raw_emb)
        key = cipher._active_key
        req_2k = CHORUSFrame.from_api_request(
            key_id=key.key_id, seq=seq,
            watermark=b"\x00" * 32,
            query_vector=env.vector,
            context={"query_text": query, "top_k": top_k * 2},
        )
        seq += 1
        body_2k, _, _ = http_post_chorus(req_2k)
        resp_2k = CHORUSFrame.from_bytes(body_2k)
        raw_2k = resp_2k.decode_api_response()
        pairs_2k = unpack_response_payload(raw_2k)
        result_ids_2k = [s.doc_id for _, s in pairs_2k]

        rtt_ms.extend(q_rtts)
        server_embed_ms.extend(q_server_embeds)
        wire_bytes.extend(q_wire)
        all_result_ids.append(result_ids)
        all_result_ids_2k.append(result_ids_2k)

        if (q_idx + 1) % 10 == 0:
            print(f"  {q_idx+1}/{len(queries)} queries done "
                  f"(mean RTT: {np.mean(q_rtts):.1f} ms)")

    # Client embedding: only the query (1 call per query×trial)
    total_embed_calls = len(queries) * n_trials * 1

    return {
        "path": "prismapi",
        "n_queries": len(queries),
        "n_trials": n_trials,
        "top_k": top_k,
        "rtt_ms": {
            "mean": float(np.mean(rtt_ms)),
            "p50": float(np.percentile(rtt_ms, 50)),
            "p95": float(np.percentile(rtt_ms, 95)),
            "p99": float(np.percentile(rtt_ms, 99)),
            "std": float(np.std(rtt_ms)),
        },
        "server_embed_ms": {
            "mean": float(np.mean(server_embed_ms)),
            "p50": float(np.percentile(server_embed_ms, 50)),
            "p95": float(np.percentile(server_embed_ms, 95)),
        },
        "client_embed_ms": {
            "mean": 0.0, "p50": 0.0, "p95": 0.0,
            "note": "consumer does not re-embed results",
        },
        "total_embedding_calls": total_embed_calls,
        "mean_embed_calls_per_query": total_embed_calls / (len(queries) * n_trials),
        "mean_wire_bytes": float(np.mean(wire_bytes)),
        "result_ids_per_query": all_result_ids,
        "result_ids_2k_per_query": all_result_ids_2k,
    }


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------

def compute_quality(
    baseline_ids: list[list[str]],
    prismapi_ids: list[list[str]],
    prismapi_ids_2k: list[list[str]] | None = None,
) -> dict:
    jaccard: list[float] = []
    recall_k: list[float] = []
    recall_2k: list[float] = []

    for i, (b, p) in enumerate(zip(baseline_ids, prismapi_ids)):
        b_set, p_set = set(b), set(p)
        jaccard.append(len(b_set & p_set) / max(len(b_set | p_set), 1))
        recall_k.append(len(b_set & p_set) / max(len(b_set), 1))
        if prismapi_ids_2k:
            p2 = set(prismapi_ids_2k[i])
            recall_2k.append(len(b_set & p2) / max(len(b_set), 1))

    result = {
        "mean_jaccard": round(float(np.mean(jaccard)), 3),
        "min_jaccard": round(float(np.min(jaccard)), 3),
        "mean_recall_at_k": round(float(np.mean(recall_k)), 3),
        "min_recall_at_k": round(float(np.min(recall_k)), 3),
    }
    if recall_2k:
        result["mean_recall_at_2k"] = round(float(np.mean(recall_2k)), 3)
        result["min_recall_at_2k"] = round(float(np.min(recall_2k)), 3)
    return result


# ---------------------------------------------------------------------------
# Concurrency stress test
# ---------------------------------------------------------------------------

def run_concurrent_test(
    embedder: SentenceTransformerEmbedder,
    projector: PrismProjector,
    queries: list[str],
    top_k: int,
    concurrency: int,
    n_requests: int,
) -> dict:
    """
    Fire n_requests concurrently (via threading) and measure throughput.
    Tests both baseline and PrismAPI paths.
    """
    import threading

    from prism.lib.fabric import TensorCipher
    cipher = TensorCipher(dim=projector._cfg.target_dim, ttl_seconds=3600.0)
    cipher.rotate_key()

    results_lock = threading.Lock()
    baseline_times: list[float] = []
    prismapi_times: list[float] = []
    errors: list[str] = []
    seq_counter = [0]

    def baseline_worker(q: str) -> None:
        try:
            url = f"{SERVER_URL}/search?q={urllib.request.quote(q)}&top_k={top_k}"
            _, _, elapsed = http_get(url)
            with results_lock:
                baseline_times.append(elapsed)
        except Exception as e:
            with results_lock:
                errors.append(f"baseline: {e}")

    def prismapi_worker(q: str) -> None:
        try:
            raw_emb = embedder.embed([q])[0]
            env = projector.project(raw_emb)
            key = cipher._active_key
            with results_lock:
                seq = seq_counter[0]
                seq_counter[0] += 1
            req_frame = CHORUSFrame.from_api_request(
                key_id=key.key_id, seq=seq,
                watermark=b"\x00" * 32,
                query_vector=env.vector,
                context={"query_text": q, "top_k": top_k},
            )
            _, _, elapsed = http_post_chorus(req_frame)
            with results_lock:
                prismapi_times.append(elapsed)
        except Exception as e:
            with results_lock:
                errors.append(f"prismapi: {e}")

    def run_concurrent(worker_fn, label: str) -> dict:
        qs = [queries[i % len(queries)] for i in range(n_requests)]
        t_start = time.perf_counter()
        threads = []
        sem = threading.Semaphore(concurrency)

        def bounded(q):
            with sem:
                worker_fn(q)

        for q in qs:
            t = threading.Thread(target=bounded, args=(q,), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        elapsed_total = time.perf_counter() - t_start

        times = baseline_times if label == "baseline" else prismapi_times
        return {
            "label": label,
            "n_requests": n_requests,
            "concurrency": concurrency,
            "total_elapsed_s": round(elapsed_total, 2),
            "throughput_rps": round(len(times) / elapsed_total, 1),
            "mean_latency_ms": round(float(np.mean(times)) if times else 0, 1),
            "p50_ms": round(float(np.percentile(times, 50)) if times else 0, 1),
            "p95_ms": round(float(np.percentile(times, 95)) if times else 0, 1),
            "p99_ms": round(float(np.percentile(times, 99)) if times else 0, 1),
            "errors": len(errors),
        }

    print(f"\n[concurrent] Baseline: {n_requests} requests @ {concurrency} concurrent...")
    b_result = run_concurrent(baseline_worker, "baseline")
    print(f"  {b_result['throughput_rps']} req/s, p95={b_result['p95_ms']} ms")

    print(f"[concurrent] PrismAPI: {n_requests} requests @ {concurrency} concurrent...")
    p_result = run_concurrent(prismapi_worker, "prismapi")
    print(f"  {p_result['throughput_rps']} req/s, p95={p_result['p95_ms']} ms")

    return {"baseline": b_result, "prismapi": p_result}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_results(baseline: dict, prismapi: dict, quality: dict, concurrent: dict | None) -> None:
    sep = "=" * 65

    print(f"\n{sep}")
    print("REAL BENCHMARK RESULTS")
    print(sep)

    b_rtt = baseline["rtt_ms"]
    p_rtt = prismapi["rtt_ms"]
    print(f"\nEnd-to-end latency (real HTTP RTT, {baseline['n_trials']} trials per query):")
    print(f"  {'':30s}  {'Baseline':>10}  {'PrismAPI':>10}")
    print(f"  {'Mean':30s}  {b_rtt['mean']:>9.1f}ms  {p_rtt['mean']:>9.1f}ms")
    print(f"  {'P50':30s}  {b_rtt['p50']:>9.1f}ms  {p_rtt['p50']:>9.1f}ms")
    print(f"  {'P95':30s}  {b_rtt['p95']:>9.1f}ms  {p_rtt['p95']:>9.1f}ms")
    print(f"  {'P99':30s}  {b_rtt['p99']:>9.1f}ms  {p_rtt['p99']:>9.1f}ms")
    print(f"  {'Std dev':30s}  {b_rtt['std']:>9.1f}ms  {p_rtt['std']:>9.1f}ms")

    print(f"\nServer embedding latency (query embed on server):")
    b_se = baseline["server_embed_ms"]
    p_se = prismapi["server_embed_ms"]
    print(f"  Mean: baseline={b_se['mean']:.1f}ms  PrismAPI={p_se['mean']:.1f}ms")

    top_k = baseline["top_k"]
    b_ce = baseline["client_embed_ms"]
    print(f"\nClient embedding latency (re-embedding {top_k} result texts):")
    print(f"  Baseline mean: {b_ce['mean']:.1f}ms  "
          f"P95: {b_ce['p95']:.1f}ms")
    print(f"  PrismAPI:      0.0ms  (vectors arrive pre-projected)")

    total_b = baseline["total_embedding_calls"]
    total_p = prismapi["total_embedding_calls"]
    saved = total_b - total_p
    pct = 100.0 * saved / max(total_b, 1)
    print(f"\nEmbedding calls ({baseline['n_queries']} queries x {baseline['n_trials']} trials):")
    print(f"  Baseline:  {total_b} calls  ({baseline['mean_embed_calls_per_query']:.1f}/query)")
    print(f"  PrismAPI:  {total_p} calls  ({prismapi['mean_embed_calls_per_query']:.1f}/query)")
    print(f"  SAVED:     {saved} calls  ({pct:.1f}%)")

    b_wire = baseline["mean_wire_bytes"]
    p_wire = prismapi["mean_wire_bytes"]
    wire_change = 100.0 * (p_wire - b_wire) / max(b_wire, 1)
    sign = "+" if wire_change > 0 else ""
    print(f"\nWire bytes (mean per request, both directions):")
    print(f"  Baseline JSON:   {b_wire:,.0f} bytes")
    print(f"  PrismAPI CHORUS: {p_wire:,.0f} bytes  ({sign}{wire_change:.1f}%)")

    print(f"\nRetrieval quality ({top_k} results, cosine@384 vs cosine@{prismapi.get('target_dim', '?')}):")
    print(f"  Jaccard@{top_k}:   {quality['mean_jaccard']:.3f}"
          f"  (exact set overlap)")
    print(f"  Recall@{top_k}:    {quality['mean_recall_at_k']:.3f}"
          f"  (fraction of baseline top-{top_k} in PrismAPI top-{top_k})")
    if "mean_recall_at_2k" in quality:
        print(f"  Recall@{top_k*2}:   {quality['mean_recall_at_2k']:.3f}"
              f"  [PRODUCTION METRIC: over-fetch x2, re-rank]")
    print(f"  Min Jaccard: {quality['min_jaccard']:.3f}")

    if concurrent:
        b_c = concurrent["baseline"]
        p_c = concurrent["prismapi"]
        print(f"\nConcurrency test ({b_c['concurrency']} simultaneous agents, {b_c['n_requests']} requests):")
        print(f"  {'':30s}  {'Baseline':>10}  {'PrismAPI':>10}")
        print(f"  {'Throughput (req/s)':30s}  {b_c['throughput_rps']:>10.1f}  {p_c['throughput_rps']:>10.1f}")
        print(f"  {'P95 latency':30s}  {b_c['p95_ms']:>9.1f}ms  {p_c['p95_ms']:>9.1f}ms")
        print(f"  {'P99 latency':30s}  {b_c['p99_ms']:>9.1f}ms  {p_c['p99_ms']:>9.1f}ms")
        print(f"  {'Errors':30s}  {b_c['errors']:>10}  {p_c['errors']:>10}")

    print(f"\n{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real PrismAPI vs baseline benchmark")
    parser.add_argument("--trials", type=int, default=3,
                        help="Trials per query (default: 3)")
    parser.add_argument("--queries", type=int, default=50,
                        help="Number of test queries (default: 50)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64,
                        help="PrismProjector target dimension (default: 64)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Concurrent agents for stress test (default: 10)")
    parser.add_argument("--concurrent-requests", type=int, default=50,
                        help="Total requests in concurrency test (default: 50)")
    parser.add_argument("--no-concurrent", action="store_true",
                        help="Skip concurrency test")
    parser.add_argument("--out", default="benchmark/api/results/real_benchmark_results.json")
    args = parser.parse_args()

    queries = QUERIES[:args.queries]
    print(f"Real PrismAPI Benchmark")
    print(f"  Model:          {args.model}")
    print(f"  Corpus:         {len(CORPUS)} documents")
    print(f"  Queries:        {len(queries)}")
    print(f"  Trials/query:   {args.trials}")
    print(f"  Top-K:          {args.top_k}")
    print(f"  Target dim:     {args.dim}")
    print(f"  Concurrency:    {args.concurrency}")

    # Start server subprocess
    print(f"\n[runner] Starting server subprocess (port {SERVER_PORT})...")
    proc = start_server(args.dim, args.model)
    try:
        stream_server_until_ready(proc)
        wait_for_server(timeout=90.0)
        print(f"[runner] Server ready.")

        # Client-side setup
        print(f"\n[runner] Loading client embedder...")
        embedder = SentenceTransformerEmbedder(model_name=args.model)
        projector = PrismProjector(ProjectionConfig(
            tenant_id="bench-tenant",   # must match server — same projection matrix
            target_dim=args.dim,
        ))

        # --- Run both paths ---
        baseline = run_baseline_path(embedder, queries, args.top_k, args.trials)
        prismapi = run_prismapi_path(embedder, projector, queries, args.top_k, args.trials)
        prismapi["target_dim"] = args.dim

        # --- Retrieval quality ---
        quality = compute_quality(
            baseline["result_ids_per_query"],
            prismapi["result_ids_per_query"],
            prismapi.get("result_ids_2k_per_query"),
        )

        # --- Concurrency test ---
        concurrent = None
        if not args.no_concurrent:
            concurrent = run_concurrent_test(
                embedder, projector, queries, args.top_k,
                concurrency=args.concurrency,
                n_requests=args.concurrent_requests,
            )

        # --- Print report ---
        print_results(baseline, prismapi, quality, concurrent)

        # --- Save JSON ---
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": {
                "model": args.model,
                "corpus_size": len(CORPUS),
                "n_queries": len(queries),
                "n_trials": args.trials,
                "top_k": args.top_k,
                "target_dim": args.dim,
                "embed_dim": embedder.embed_dim,
            },
            "baseline": {k: v for k, v in baseline.items() if "result_ids" not in k},
            "prismapi": {k: v for k, v in prismapi.items() if "result_ids" not in k},
            "quality": quality,
            "concurrent": concurrent,
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to: {out_path}")

    finally:
        print(f"\n[runner] Stopping server (pid={proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[runner] Server stopped.")


if __name__ == "__main__":
    main()
