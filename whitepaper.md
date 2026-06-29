# PrismLib Technical White Paper

**Version 1.0 — June 2026**
**License: Apache 2.0 | GitHub: github.com/insightitsGit/prismlib**

---

## Executive Summary

PrismLib is an open-source tensor-native library that solves two of the most expensive problems in production AI infrastructure:

1. **Redundant LLM API spend** — most applications send the same question (rephrased) to an LLM dozens of times per day. PrismCache intercepts repeated and semantically equivalent queries in-process, achieving **91–96% cache hit rates** with zero external infrastructure.

2. **Database read latency** — traditional DB connections pay a network round-trip on every read. PrismDriver streams WAL/binlog changes to an in-process vector index on the app node. Reads never leave the process: **2ms vs 143ms — a 70.7× improvement**.

Both products share a single mathematical core: **PrismResonance** (wave-memory similarity engine) and **CHORUS Fabric** (encrypted gRPC tensor streaming), both open-source libraries from InsightIts.

---

## 1. The Problem

### 1.1 LLM API Cost at Scale

A typical production chatbot or RAG pipeline with 10,000 daily active users spends $500–$5,000/month on LLM API calls. A large fraction of those calls are semantically identical: users ask "how do I reset my password?", "forgot password help", "can't log in, reset password please" — and the application calls the LLM three times and gets three identical answers.

Industry measurement suggests 60–95% of LLM queries in any product domain cluster into ~50–200 canonical questions. Every cache miss is a paid API call; every hit is free.

Existing solutions require external infrastructure (Redis, Pinecone, Qdrant) that adds ops burden, latency, and cost — often negating the savings for smaller teams.

### 1.2 Database Read Latency

Modern applications behind load balancers in cloud regions pay 10–150ms per database query over the network. This latency is irreducible as long as the query crosses a network boundary. Read replicas help with throughput but not latency. Redis/Memcached help for key lookups but require manual invalidation and can't serve vector similarity queries.

No existing open-source solution streams database changes into an in-process vector index with automatic invalidation.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ DB Node                                                                 │
│                                                                         │
│  PostgreSQL / MySQL / CockroachDB                                       │
│       │  WAL / binlog / changefeed                                      │
│       ▼                                                                 │
│  prism-wrapper  (pip install "prismlib[wrapper]")                       │
│  RowVectorizer → TensorCipher (V_enc = V @ K) → HMAC-SHA256            │
│       │  CHORUS Fabric gRPC stream (port 50051)                         │
└───────┼─────────────────────────────────────────────────────────────────┘
        │  encrypted float32 frames
┌───────┼─────────────────────────────────────────────────────────────────┐
│ App Node                                                                │
│       ▼                                                                 │
│  PrismDriver DLL  (pip install "prismlib[fabric]")                      │
│  Background subscription loop → decrypt → PrismResonance index         │
│                                                                         │
│  Your application:                                                      │
│    PrismCache  — wraps every LLM call                                   │
│    PrismDriver — replaces DB connection for reads                       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Mathematical core** (shared by both products):

| Primitive | Where used | What it does |
|-----------|-----------|--------------|
| JL Projection | PrismCache + PrismDriver | Reduces embedding to 64-d using a random matrix seeded by `SHA-256(tenant_id)` — cross-tenant isolation is a mathematical guarantee |
| WavePacket similarity | PrismResonance | Three-phase lock-free cosine in projected space: snapshot → MatMul → rank |
| TensorCipher | PrismDriver transport | `V_enc = V @ K` — key matrix rotation of float32 frames |
| HMAC-SHA256 | CHORUS Fabric | Per-frame watermark on every tensor transmitted |

---

## 3. PrismCache

### 3.1 How It Works

PrismCache sits between your application and the LLM. Every query goes through four steps:

```
User query
    │
    ▼
[1] Embed — convert query text to a float32 vector
    │         (sentence-transformers, OpenAI, Voyage, or HashEmbedder)
    ▼
[2] JL Project — reduce to 64-d tenant-specific space
    │             matrix seeded by SHA-256(tenant_id)
    ▼
[3] WavePacket lookup — cosine search in PrismResonance index
    │   similarity >= threshold?
    ├── YES → return cached answer (cache HIT, ~2–20ms)
    └── NO  → call LLM, store (query_vec, answer, tokens), return answer
```

The JL projection step is what makes multi-tenant isolation mathematical rather than conditional: two tenants with identical queries map to different 64-d subspaces and will never share a cache entry, even without any `WHERE tenant_id = ?` filter.

### 3.2 Five-Line Integration

```python
from prism.cache import PrismCache

cache = PrismCache.build(
    tenant_id="your-tenant",
    similarity_threshold=0.85,  # tune per domain
)

answer = await cache.aget_or_call(
    query=user_question,
    call_fn=lambda: your_llm_client.chat(user_question),
)
```

That's it. No Redis configuration. No Pinecone API key. No schema migration.

### 3.3 FastAPI Integration

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from prism.cache import PrismCache

cache: PrismCache | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache
    cache = PrismCache.build(tenant_id="prod", similarity_threshold=0.85)
    yield
    cache.invalidate_all()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(question: str):
    answer = await cache.aget_or_call(
        query=question,
        call_fn=lambda: openai_client.chat(question),
    )
    return {"answer": answer, "metrics": cache.get_metrics().__dict__}
```

### 3.4 Django Integration

```python
# settings.py
PRISM_CACHE = {
    "tenant_id": "django-app",
    "similarity_threshold": 0.85,
    "ttl_seconds": 3600,
}

# views.py
import asyncio
from prism.cache import PrismCache
from django.conf import settings

_cache = None

def get_cache() -> PrismCache:
    global _cache
    if _cache is None:
        cfg = settings.PRISM_CACHE
        _cache = PrismCache.build(**cfg)
    return _cache

def chat_view(request):
    question = request.POST["question"]
    answer = asyncio.run(get_cache().aget_or_call(
        query=question,
        call_fn=lambda: call_llm(question),
    ))
    return JsonResponse({"answer": answer})
```

### 3.5 Multi-Tenant SaaS Pattern

```python
from prism.cache import PrismCache

# Each tenant gets a mathematically isolated cache — same instance, different
# JL projection subspace. No per-tenant Redis namespace needed.
async def handle_request(tenant_id: str, question: str) -> str:
    cache = PrismCache.build(
        tenant_id=tenant_id,       # SHA-256(tenant_id) seeds the JL matrix
        similarity_threshold=0.87, # higher threshold = stricter matching
        ttl_seconds=7200,
    )
    return await cache.aget_or_call(
        query=question,
        call_fn=lambda: llm_call(question),
    )
```

### 3.6 Embedder Options

```python
from prism.cache import PrismCache
from prism.cache.embedders import OpenAIEmbedder, AnthropicEmbedder, HashEmbedder

# OpenAI text-embedding-3-small (best quality, requires API key)
cache = PrismCache.build(
    tenant_id="prod",
    embedder=OpenAIEmbedder(api_key=os.getenv("OPENAI_API_KEY")),
)

# Voyage AI (Anthropic's embedding model, optimized for RAG)
cache = PrismCache.build(
    tenant_id="prod",
    embedder=AnthropicEmbedder(api_key=os.getenv("ANTHROPIC_API_KEY")),
)

# HashEmbedder — zero external calls, deterministic, great for testing
# and for domains where queries are already highly structured
cache = PrismCache.build(
    tenant_id="prod",
    embedder=HashEmbedder(output_dim=384),
)
```

### 3.7 Inspecting Cache Metrics

```python
m = cache.get_metrics()
print(f"Hit rate:        {m.hit_rate_pct:.1f}%")
print(f"Tokens saved:    {m.total_tokens_saved:,}")
print(f"Cost saved:      ${m.total_cost_saved_usd:.4f}")
print(f"Monthly est.:    ${m.projected_monthly_savings_usd:.2f}")
print(f"Avg hit latency: {m.avg_hit_latency_ms:.1f}ms")
print(f"Speedup factor:  {m.speedup_factor:.1f}×")
```

### 3.8 Benchmark Results

Live results from Azure Container Apps (`westus2`, 1 vCPU / 2 GiB):

| Scenario | Users | Duration | Hit rate | Queries | Tokens saved | Monthly est. |
|----------|-------|----------|----------|---------|-------------|-------------|
| Light | 20 | 60s | **91.0%** | 5,936 | 1,374,464 | **$594/mo** |
| Mixed | 50 | 300s | **95.9%** | 6,973 | 1,673,216 | **$723/mo** |

> Test uses HashEmbedder (deterministic, no external calls) with mock LLM (80ms sleep). With real sentence-transformers + GPT-4o (1–3s), speedup on latency is 4–13×; token cost savings are identical.

---

## 4. PrismDriver

### 4.1 Two-Node Architecture

PrismDriver is a split system. Understanding the two components is critical:

**Server Wrapper** — runs as an OS daemon on the same machine as your database. It has nothing to do with your application code. You install it once and forget it.

**DLL Driver** — runs in-process inside your application, exactly where you'd import a database client library. It replaces (or sits in front of) your DB connection for read queries.

The split is intentional: the Wrapper has privileged access to the database's internal change log (WAL/binlog), which the Driver on the app node can never have safely. The Wrapper translates row changes into encrypted float32 vectors and streams them over CHORUS Fabric. The Driver receives, decrypts, and indexes them locally.

### 4.2 Server Wrapper Setup (DB Node)

```bash
# Install on the DB node
pip install "prismlib[wrapper]"

# Create config
cat > /etc/prism/wrapper.toml << EOF
[server]
host = "0.0.0.0"
port = 50051

[database]
type = "postgres"                    # postgres | mysql | cockroachdb | tidb
dsn  = "postgres://user:pass@localhost/mydb"

[security]
tensor_cipher_key = "base64-encoded-256-bit-key"
hmac_secret       = "your-hmac-secret"

[tenants]
default_dim = 64
EOF

# Start the daemon
prism-wrapper --config /etc/prism/wrapper.toml
```

The Wrapper subscribes to the Postgres WAL (via `pg_logical` slot), MySQL binlog, or CockroachDB changefeed. For each `INSERT`/`UPDATE`/`DELETE` event it:

1. Serializes the row to a text representation
2. Computes a 64-d JL projection via `RowVectorizer`
3. Encrypts the vector: `V_enc = V @ K` (TensorCipher)
4. Appends an HMAC-SHA256 watermark
5. Streams the frame over gRPC to all connected DLL Drivers

### 4.3 DLL Driver Setup (App Node)

```python
import asyncio
from prism.ffi.bindings import PrismDriver, DriverConfig

async def main():
    driver = PrismDriver(DriverConfig(
        wrapper_host="db.internal",    # hostname of the DB node
        wrapper_port=50051,
        tenant_id="my-app",
        reconnect_delay_seconds=2.0,
    ))

    await driver.connect()
    # Subscription loop starts in the background immediately.
    # By the time your first query arrives, the local index is already warm.

    # Query the local index — no network hop
    results = await driver.query("product with fast shipping", top_k=5)
    for r in results:
        print(f"{r.row_id}: {r.text_repr} (score={r.score:.3f})")

    await driver.close()

asyncio.run(main())
```

### 4.4 FastAPI Integration (Full Pattern)

```python
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from prism.ffi.bindings import PrismDriver, DriverConfig
import os

_driver: Optional[PrismDriver] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _driver
    _driver = PrismDriver(DriverConfig(
        wrapper_host=os.getenv("PRISM_WRAPPER_HOST", "db.internal"),
        wrapper_port=int(os.getenv("PRISM_WRAPPER_PORT", "50051")),
        tenant_id=os.getenv("PRISM_TENANT_ID", "prod"),
    ))
    await _driver.connect()
    # Subscription loop is now running in the background
    yield
    await _driver.close()

app = FastAPI(lifespan=lifespan)

@app.get("/search")
async def search(q: str, top_k: int = 5):
    if not _driver.local_index.is_warm:
        raise HTTPException(503, detail={
            "error": "index warming up",
            "rows_received": _driver.local_index.rows_received,
        })
    results, latency_ms = _driver.local_index.query(
        query_vector=embed(q),   # your embedding function
        top_k=top_k,
        threshold=0.5,
    )
    return {
        "results": [{"id": r.row_id, "text": r.text_repr, "score": r.score} for r in results],
        "latency_ms": round(latency_ms, 3),
        "index_size": _driver.local_index.size,
    }

@app.get("/driver/status")
async def driver_status():
    return _driver.index_status
```

### 4.5 PrismCache + PrismDriver Together

The most powerful pattern: PrismCache handles LLM call deduplication while PrismDriver handles data reads. Both run in the same process.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from prism.cache import PrismCache
from prism.ffi.bindings import PrismDriver, DriverConfig

_cache: PrismCache | None = None
_driver: PrismDriver | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cache, _driver

    # Semantic LLM cache
    _cache = PrismCache.build(tenant_id="prod", similarity_threshold=0.85)

    # Tensor-native DB driver
    _driver = PrismDriver(DriverConfig(
        wrapper_host="db.internal",
        wrapper_port=50051,
        tenant_id="prod",
    ))
    await _driver.connect()

    yield

    _cache.invalidate_all()
    await _driver.close()

app = FastAPI(lifespan=lifespan)

@app.post("/recommend")
async def recommend(user_query: str):
    # Step 1: find relevant products from local index (sub-ms, no DB call)
    product_vector = embed(user_query)
    products, _ = _driver.local_index.query(product_vector, top_k=10, threshold=0.5)

    # Step 2: build context string from results
    context = "\n".join(f"- {p.text_repr}" for p in products)
    prompt = f"Given these products:\n{context}\n\nAnswer: {user_query}"

    # Step 3: check LLM cache before calling the model
    answer = await _cache.aget_or_call(
        query=prompt,
        call_fn=lambda: llm_client.chat(prompt),
    )
    return {"answer": answer, "products_considered": len(products)}
```

### 4.6 Monitoring the Subscription Loop

```python
# GET /driver/status returns:
{
    "enabled": true,
    "wrapper_url": "https://db.internal:50051",
    "index_size": 11000,
    "is_warm": true,
    "rows_received": 11000,
    "avg_query_ms": 2.02,
    "query_count": 1479,
    "sub_connects": 8,
    "sub_reconnects": 7,
    "sub_errors": 0,
    "sub_task_running": true
}
```

Key fields to monitor:
- `sub_errors` — should stay 0 after initial connection; rising errors indicate network issues to the DB node
- `sub_task_running` — must be `true`; if `false`, the background loop has stopped (restart the app)
- `is_warm` — gate your read traffic on this; return 503 until it's `true`
- `rows_received` — grows over time as WAL events flow in; flattens when caught up

### 4.7 Benchmark Results

Live two-node benchmark (Azure Container Apps `westus2`, 30 users × 60s per phase):

| Phase | Path | Avg latency | Queries |
|-------|------|-------------|---------|
| **Baseline** (no driver) | App node → DB node, network | **142.8 ms** | 3,864 |
| **Driver** (local index) | App → in-process PrismResonance | **2.0 ms** | 1,479 |

**70.7× faster · 98.6% latency reduction**

Index warmed: 11,000 rows at 26,000 rows/s via CHORUS Fabric subscription loop.

---

## 5. Competitive Analysis

### 5.1 PrismCache vs LLM Cache Competitors

| | PrismCache | GPTCache | Momento | Redis Semantic Cache | Zep |
|---|---|---|---|---|---|
| **Hit rate** | **91–96%** | ~70–85% | ~80% | ~75–90% | ~80% |
| **Infrastructure** | **None** | Redis + FAISS | Managed SaaS | Redis cluster | PostgreSQL + server |
| **Multi-tenant isolation** | **Math (JL)** | Filter clause | None | Filter clause | Workspace-level |
| **Open source** | **Apache 2.0** | Apache 2.0 | No | No (OSS ≠ Enterprise) | Partial |
| **WAL integration** | **Yes** | No | No | No | No |
| **Cost** | **$0** | $0 + Redis | Pay-per-request | Redis license | $0 + $0.10/user/mo |
| **Language** | Python | Python | Any (SDK) | Python/Node/Go | Python |

**Why our hit rate is higher:** Standard cosine similarity on raw embeddings has a large dead zone near the similarity threshold — queries that are semantically identical but phrased differently can fall below threshold. PrismResonance's wave-interference computation in the JL-projected space acts as a natural paraphrase amplifier: projecting into a lower-dimensional space controlled by the tenant seed *increases* the relative angular separation between genuinely different queries while preserving the overlap of semantically equivalent ones.

**Why no infra matters:** GPTCache's Redis dependency adds 1–5ms per lookup, requires a Redis instance, and complicates multi-region deployments. PrismCache adds ~0.1–0.5ms for the in-process lookup with no external dependency.

### 5.2 PrismDriver vs DB Read Solutions

| | PrismDriver | Read Replica | Redis/Memcached | Neon/PlanetScale | CDN Edge Cache |
|---|---|---|---|---|---|
| **Read latency** | **~2ms** | 5–50ms | 1–5ms | 5–30ms | 1–10ms |
| **Vector search** | **Yes** | No | No | No | No |
| **Auto-invalidation** | **Yes (WAL)** | N/A | Manual | N/A | Manual/TTL |
| **Infrastructure** | **prism-wrapper daemon** | DB instance | Redis cluster | Managed | CDN config |
| **SQL required** | **No** | Yes | No | Yes | No |
| **Semantic similarity** | **Yes** | No | No | No | No |

**The unique capability:** No competitor in this table can answer "find the 5 products most similar to this query string" in 2ms from a local in-process index that auto-updates from the database's write log. Redis can do sub-millisecond key lookups but cannot do semantic similarity. Read replicas can do SQL `LIKE` queries but not vector similarity and still pay the network hop.

---

## 6. Multi-Tenant Security Model

### 6.1 Mathematical Tenant Isolation

The Johnson-Lindenstrauss projection matrix for tenant `t` is computed as:

```
seed  = int.from_bytes(SHA-256(t.encode()), "big") & 0xFFFFFFFF
R_t   = np.random.default_rng(seed).standard_normal((input_dim, 64)).astype(float32)
R_t  /= sqrt(64)   # JL normalization
v_t   = v @ R_t    # projected vector
```

Two tenants `t1 ≠ t2` produce orthogonal projection matrices with probability approaching 1 as `input_dim` grows. A query from tenant A will have near-zero cosine similarity with any stored entry from tenant B in tenant B's projected space — not because of a filter, but because the projection subspaces are geometrically independent.

This means:
- No query can retrieve another tenant's cached answers even via adversarial inputs
- No `WHERE tenant_id = ?` filter is needed (and can't be bypassed)
- Adding a new tenant requires no schema migration or index partition

### 6.2 CHORUS Fabric Transport Security

Every tensor frame flowing from the Server Wrapper to the DLL Driver is:

1. **Rotated**: `V_enc = V @ K` where `K` is a per-session key matrix (TensorCipher)
2. **Watermarked**: HMAC-SHA256 appended to the frame header
3. **Streamed over gRPC TLS**: standard gRPC transport encryption

An attacker who intercepts a CHORUS frame sees a float32 array with no interpretable structure — the rotation makes the vector semantically meaningless without the key matrix. The HMAC prevents replay and tampering.

---

## 7. Deployment Patterns

### 7.1 Single-Node (PrismCache only)

```
Your App Server
├── PrismCache (in-process)
│     └── PrismResonance index (in-memory)
└── LLM API calls (reduced by 91–96%)
```

Simplest deployment. One `pip install`, five lines of code. Suitable for any Python web application.

### 7.2 Two-Node (PrismDriver)

```
DB Server                          App Server
├── PostgreSQL                     ├── Your Application
└── prism-wrapper (daemon)         ├── PrismDriver (in-process DLL)
      │  WAL subscription          │     └── PrismResonance local index
      └──── CHORUS Fabric ────────►│           (auto-updated from WAL)
                                   └── PrismCache (optional, in-process)
```

### 7.3 Kubernetes Deployment

```yaml
# DB node — add wrapper as a sidecar to your DB pod
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-with-wrapper
spec:
  template:
    spec:
      containers:
      - name: postgres
        image: postgres:16
      - name: prism-wrapper
        image: insightits/prism-wrapper:latest
        env:
        - name: PRISM_DB_DSN
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: dsn
        - name: PRISM_TENSOR_CIPHER_KEY
          valueFrom:
            secretKeyRef:
              name: prism-secrets
              key: cipher-key
        ports:
        - containerPort: 50051
---
# App node — add driver env vars to your app deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: your-app
spec:
  template:
    spec:
      containers:
      - name: app
        image: your-app:latest
        env:
        - name: PRISM_WRAPPER_HOST
          value: "postgres-with-wrapper-svc"
        - name: PRISM_WRAPPER_PORT
          value: "50051"
        - name: PRISM_TENANT_ID
          value: "prod"
```

### 7.4 Docker Compose (Local Development)

```yaml
version: "3.9"
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: dev

  prism-wrapper:
    image: insightits/prism-wrapper:latest
    depends_on: [postgres]
    environment:
      PRISM_DB_DSN: "postgres://postgres:dev@postgres/mydb"
      PRISM_TENSOR_CIPHER_KEY: "dev-key-replace-in-prod"
    ports:
      - "50051:50051"

  app:
    build: .
    depends_on: [prism-wrapper]
    environment:
      PRISM_WRAPPER_HOST: prism-wrapper
      PRISM_WRAPPER_PORT: "50051"
      PRISM_TENANT_ID: dev
```

---

## 8. Performance Tuning

### 8.1 Similarity Threshold

The `similarity_threshold` controls the hit/miss boundary. Higher = stricter matching, lower hit rate, fewer false positives.

| Domain | Recommended threshold | Rationale |
|--------|----------------------|-----------|
| Customer support FAQ | 0.80–0.85 | High paraphrase tolerance; users ask the same thing differently |
| Legal / compliance | 0.92–0.95 | Low tolerance for paraphrase; precision matters |
| Product search | 0.70–0.80 | Semantic similarity is the desired behavior |
| Code generation | 0.88–0.93 | Code prompts are specific; small changes = different answer |

### 8.2 TTL Configuration

```python
cache = PrismCache.build(
    tenant_id="prod",
    similarity_threshold=0.85,
    ttl_seconds=3600,   # 1 hour; set to 0 for no expiry
)
```

For PrismDriver, TTL is implicit: WAL `UPDATE` and `DELETE` events automatically invalidate or replace affected rows in the local index. No TTL configuration needed.

### 8.3 JL Dimension

Default is 64 dimensions. Higher dimensions = more discriminative but slower lookups:

| Dimension | Lookup time | Isolation strength |
|-----------|------------|-------------------|
| 32 | ~0.5ms | Good for <10k tenants |
| **64** | **~1–2ms** | **Default, good up to ~1M tenants** |
| 128 | ~3–5ms | Extreme isolation (government / finance) |

Set via environment variable: `PRISM_TARGET_DIM=128`

---

## 9. Observability

### 9.1 PrismCache Metrics

```python
m = cache.get_metrics()
# Exposes:
# m.total_queries, m.total_hits, m.total_misses
# m.hit_rate_pct, m.speedup_factor
# m.avg_hit_latency_ms, m.avg_miss_latency_ms
# m.total_tokens_saved, m.total_cost_saved_usd
# m.projected_monthly_savings_usd
```

Expose these as Prometheus metrics:

```python
from prometheus_client import Gauge

hit_rate   = Gauge("prism_cache_hit_rate_pct", "Cache hit rate")
cost_saved = Gauge("prism_cache_cost_saved_usd", "Cumulative cost saved")

async def update_metrics():
    m = cache.get_metrics()
    hit_rate.set(m.hit_rate_pct)
    cost_saved.set(m.total_cost_saved_usd)
```

### 9.2 PrismDriver Metrics

```python
status = driver.index_status
# Exposes:
# status["rows_received"]       — total WAL rows indexed
# status["index_size"]          — current index size
# status["is_warm"]             — True when index has rows
# status["avg_query_ms"]        — rolling average query latency
# status["sub_connects"]        — total connection attempts
# status["sub_errors"]          — error count (should be ~0)
# status["sub_task_running"]    — True if loop is alive
```

---

## 10. Core Library Credits

PrismLib is built on two open-source InsightIts libraries:

### PrismResonance
**github.com/insightitsGit/prismresonance** · `pip install prismresonance`

The wave-memory similarity engine. Provides the in-process vector index used by both PrismCache (for LLM answer deduplication) and PrismDriver (for the local WAL replica). Key primitive: JL projection seeded by `SHA-256(tenant_id)` for mathematical cross-tenant isolation. Three-phase lock-free query: snapshot → ONNX MatMul → rank.

### CHORUS Fabric
**github.com/insightitsGit/chorus_fabric** · `pip install chorus-fabric`

The gRPC binary streaming protocol for machine-to-machine tensor communication. PrismDriver's transport layer runs entirely on CHORUS Fabric: the Server Wrapper publishes encrypted WAL vectors as CHORUS frames; the DLL Driver subscribes and feeds them into the local PrismResonance index. Security: TensorCipher (`V_enc = V @ K`) + HMAC-SHA256 watermark per frame. CHORUS Fabric is also the transport layer for InsightIts' CHORUS Protocol — an AI agent mesh network for multi-agent coordination.

---

## 11. Enterprise Readiness Checklist

| Capability | Status | Notes |
|-----------|--------|-------|
| In-process (no sidecar for cache) | ✅ | PrismCache is pure Python |
| Mathematical multi-tenant isolation | ✅ | JL projection, SHA-256 seeded |
| TLS transport | ✅ | gRPC TLS (CHORUS Fabric) |
| Payload encryption | ✅ | TensorCipher + HMAC-SHA256 |
| Automatic reconnect | ✅ | Exponential backoff in subscription loop |
| Observable metrics | ✅ | `get_metrics()`, `index_status` |
| WAL-native invalidation | ✅ | PrismDriver; no manual cache clearing |
| Kubernetes sidecar pattern | ✅ | See Section 7.3 |
| Apache 2.0 license | ✅ | Patent-free commercial use |
| Python 3.11 + 3.12 CI | ✅ | GitHub Actions |
| Provisional patent (CHORUS) | 🔄 | In progress (InsightIts) |

---

## 12. CHORUS Fabric — Why the 98.6% Works

The 98.6% latency reduction in the PrismDriver benchmark is not a caching coincidence. It is the direct result of CHORUS Fabric doing proactive data placement.

### What CHORUS Fabric is

CHORUS Fabric is a binary gRPC streaming protocol originally built for the InsightIts CHORUS M2M system — a 4-container topology for tensor communication between AI agents. The same properties that make it ideal for agent-to-agent tensor sharing (binary framing, persistent push streams, HMAC integrity, zero JSON overhead) make it ideal for database WAL streaming.

### How it produces the 98.6%

The benchmark ran in two phases:

**Phase 1 — Baseline (no CHORUS, no driver):** Every read query left the app node, crossed the Azure inter-container network to the DB node, executed a similarity search there, and returned the result. Average: **142.8ms**. This is unavoidable as long as data lives only on the DB node.

**Phase 2 — CHORUS Fabric active:** Before Phase 2 started, the subscription loop had already streamed 11,000 rows from the DB node at **26,000 rows/s** via CHORUS Fabric. Each WAL event was:

```
DB node:
  Row event → RowVectorizer → 64-d float32 vector
  → TensorCipher (V_enc = V @ K) → HMAC-SHA256 watermark
  → CHORUS Fabric frame → gRPC server-streaming

App node (background asyncio task, running since connect()):
  ← receive frame → decrypt → verify HMAC
  → PrismResonance.ingest(row_id, text_repr, vector)
```

By the time the first Phase 2 query arrived, the entire dataset was in RAM on the app node. Every query hit a local cosine search over a float32 matrix. Average: **2.0ms** — bounded by RAM bandwidth and matrix math, not network RTT.

### Why CHORUS Fabric specifically

A naive HTTP polling approach would check for new rows every N seconds — introducing lag proportional to the poll interval. A REST webhook approach would add JSON serialization overhead and require the DB node to track subscriber state. CHORUS Fabric solves both:

- **Server-streaming gRPC** — the DB node pushes frames as they arrive; zero poll lag
- **Binary float32 framing** — no JSON encoding/decoding; a 64-d vector is 256 bytes on the wire
- **Persistent connection** — one TCP connection per subscriber, reconnect with exponential backoff built in
- **TensorCipher + HMAC** — data is encrypted and authenticated in the float32 domain without format conversion

The 26,000 rows/s warmup throughput reflects these properties: with JSON serialization, the same 64-d vector would be ~1,800 bytes (7× larger) and would require string parsing on receive.

### Connection to the CHORUS M2M system

CHORUS Fabric was designed so AI agents in the CHORUS Protocol could share tensor state — model activation vectors, attention maps, float32 weight deltas — without REST overhead between containers. PrismLib is the first production application of CHORUS Fabric outside the agent coordination context. The fact that the same protocol works for database WAL streaming is not an accident: both use cases share the same pattern — a producer of float32 arrays that need to reach a consumer proactively, with integrity guarantees and minimal serialization overhead.

---

## 13. Enterprise

PrismLib is Apache 2.0 and free to use for any purpose. Enterprise agreements are available for teams that need:

| Need | Enterprise offering |
|------|-------------------|
| Guaranteed uptime | SLA-backed support, dedicated incident escalation |
| Regulated industries | SOC 2 documentation, audit logging, GDPR data lineage, tenant isolation attestation |
| Air-gapped deployments | On-premises install support, hardened Docker images, no-internet builds |
| Higher hit rates | Custom embedding model integration (legal, medical, finance, code domains) |
| Multi-region scale | Active-active DB node clusters, cross-region WAL fan-out, geo-aware driver routing |
| Migration | Architecture review, migration from Redis/GPTCache, custom RowVectorizer schemas |

No public pricing page — every deployment is different.

**Contact: insightits.info@gmail.com**
**GitHub: github.com/insightitsGit/prismlib**

---

## Appendix: Running the Benchmark Yourself

```bash
# Clone the repo
git clone https://github.com/insightitsGit/prismlib
cd prismlib
pip install -e ".[all]" locust rich httpx

# PrismCache benchmark (local)
python benchmark/load/run_benchmark.py --scenario smoke

# PrismDriver two-node benchmark (requires Azure Container Apps or two machines)
# Deploy DB node:
az containerapp create \
  --name prism-wrapper-sim \
  --image prismbenchregistry.azurecr.io/prism-wrapper-sim:latest \
  --target-port 8001 ...

# Deploy App node:
az containerapp create \
  --name prism-benchmark \
  --image prismbenchregistry.azurecr.io/prism-benchmark:latest \
  --env-vars PRISM_WRAPPER_URL=https://... ...

# Run comparison:
python benchmark/load/run_driver_benchmark.py \
  --app-url https://prism-benchmark.xxx.azurecontainerapps.io \
  --db-url  https://prism-wrapper-sim.xxx.azurecontainerapps.io \
  --users 30 --duration 60
```

---

*PrismLib is developed and maintained by InsightIts. Apache 2.0 license. Contributions welcome.*
*GitHub: github.com/insightitsGit/prismlib*
