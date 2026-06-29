"""
Locust load test scenarios for PrismCache benchmark.

Scenarios:
  WarmUser    — 70% hits (paraphrased questions from seeded clusters)
  ColdUser    — 100% misses (unique questions never seen before)
  MixedUser   — realistic 70/30 hit/miss ratio
  BurstUser   — rapid-fire identical queries (stress test dedup)

Run locally:
  locust -f benchmark/load/locustfile.py --host http://localhost:8000 \
    --users 50 --spawn-rate 10 --run-time 5m --headless

Run against Azure:
  locust -f benchmark/load/locustfile.py \
    --host https://prism-benchmark.azurecontainerapps.io \
    --users 200 --spawn-rate 20 --run-time 10m --headless \
    --csv results/run_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import random
import time

from locust import HttpUser, task, between, events
from benchmark.load.seed_data import get_load_questions

# Pre-generate question pools once at import time
_HIT_QUESTIONS  = get_load_questions(5000, hit_ratio=1.0)
_MISS_QUESTIONS = [
    f"Novel question {i}: explain the technical details of {topic}"
    for i, topic in enumerate([
        "blockchain consensus algorithms",
        "Kubernetes horizontal pod autoscaling",
        "TCP/IP three-way handshake",
        "RSA key generation",
        "B-tree index structure",
        "LSTM gradient flow",
        "Transformer attention mechanism",
        "gRPC bidirectional streaming",
        "ClickHouse columnar storage",
        "ZooKeeper leader election",
    ] * 500)
]


class WarmUser(HttpUser):
    """
    Simulates a user whose queries are mostly cached.
    Expected: >70% hit rate, <5ms P95 latency on hits.
    """
    wait_time = between(0.01, 0.1)
    weight    = 3

    @task(8)
    def query_hit(self):
        q = random.choice(_HIT_QUESTIONS)
        self.client.post("/query", json={"question": q, "expected_tokens": 256})

    @task(2)
    def query_miss(self):
        q = random.choice(_MISS_QUESTIONS)
        self.client.post("/query", json={"question": q, "expected_tokens": 256})

    @task(1)
    def get_metrics(self):
        self.client.get("/metrics")


class ColdUser(HttpUser):
    """
    Simulates a user sending only novel queries (all cache misses).
    Useful to measure raw LLM throughput / baseline latency.
    """
    wait_time = between(0.1, 0.5)
    weight    = 1

    _counter = 0

    @task
    def query_cold(self):
        ColdUser._counter += 1
        q = f"Cold unique question {ColdUser._counter} about topic {ColdUser._counter % 50}"
        self.client.post("/query", json={"question": q, "expected_tokens": 128})


class MixedUser(HttpUser):
    """
    Realistic production mix: 70% repeated topics, 30% novel.
    """
    wait_time = between(0.05, 0.3)
    weight    = 5

    @task(7)
    def query_warm(self):
        q = random.choice(_HIT_QUESTIONS)
        self.client.post("/query", json={"question": q})

    @task(3)
    def query_novel(self):
        q = random.choice(_MISS_QUESTIONS)
        self.client.post("/query", json={"question": q})

    @task(1)
    def batch_query(self):
        questions = random.sample(_HIT_QUESTIONS, k=min(10, len(_HIT_QUESTIONS)))
        self.client.post("/query/batch", json={"questions": questions})


class BurstUser(HttpUser):
    """
    Sends the same question repeatedly — tests dedup / thread safety.
    """
    wait_time = between(0.001, 0.01)
    weight    = 1

    BURST_QUESTION = "What is your return policy?"

    @task
    def burst_same_query(self):
        self.client.post("/query", json={"question": self.BURST_QUESTION})


# ---------------------------------------------------------------------------
# Custom event listeners — capture per-request metrics
# ---------------------------------------------------------------------------

_results: list[dict] = []


@events.request.add_listener
def on_request(
    request_type, name, response_time, response_length,
    exception, context, **kwargs
):
    _results.append({
        "type":            request_type,
        "name":            name,
        "response_time_ms": response_time,
        "success":         exception is None,
        "ts":              time.time(),
    })


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    if not _results:
        return

    hits = [r for r in _results if r["name"] == "/query" and r["success"]]
    if hits:
        times = sorted(r["response_time_ms"] for r in hits)
        n = len(times)
        print("\n── PrismCache Load Test Results ──────────────────────────")
        print(f"  Total requests : {len(_results)}")
        print(f"  /query count   : {n}")
        print(f"  P50 latency    : {times[int(n * 0.50)]:.1f}ms")
        print(f"  P95 latency    : {times[int(n * 0.95)]:.1f}ms")
        print(f"  P99 latency    : {times[int(n * 0.99)]:.1f}ms")
        print(f"  Max latency    : {times[-1]:.1f}ms")
        print(f"  Error rate     : {sum(1 for r in _results if not r['success']) / len(_results) * 100:.1f}%")
        print("──────────────────────────────────────────────────────────\n")
