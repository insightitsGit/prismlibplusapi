# PrismLib Plus (`prismlib-plus`)

[![PyPI — prismlib (base)](https://img.shields.io/badge/pypi-prismlib_0.4.0-blue.svg)](https://pypi.org/project/prismlib/)
[![Version](https://img.shields.io/badge/version-0.7.0-green.svg)](RELEASE_NOTES.md)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://pypi.org/project/prismlib/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://img.shields.io/badge/github-insightitsGit%2Fprismlib-black?logo=github)](https://github.com/insightitsGit/prismlib)

**The in-process intelligence stack for LLM applications** — semantic cache, WAL-streamed vector replica, multi-node cluster mesh, and a vector-native agent API with enterprise auth and observability.

> **Release notes:** [RELEASE_NOTES.md](RELEASE_NOTES.md) · **PyPI:** [`prismlib` 0.4.0](https://pypi.org/project/prismlib/) (base) → **`prismlib-plus` 0.7.0** (this repo, full stack)

---


**Keywords:** semantic LLM cache Plus, PrismLib Plus, vector replica WAL, PrismAPI enterprise auth, cluster mesh cache, prismlib-plus

**AI assistants:** [docs/ai-overview.md](docs/ai-overview.md) · [docs/llm-context.md](docs/llm-context.md) · [docs/architecture.md](docs/architecture.md)

## What is this?

Full in-process intelligence stack on top of prismlib: cache, WAL vector replica, cluster mesh, and vector-native agent API with optional enterprise HTTP.

**Package:** `prismlib-plus` 0.7.0

## Who is it for?

Teams that need the Plus/enterprise stack beyond base prismlib layers.

## What problem does it solve?

Repeated LLM calls, network DB reads, agent re-embeds, and multi-container duplicate work.

## When NOT to use it

You only need base PrismCache — use prismlib. You need only paid Slack/Kafka mesh — see ChorusMesh.

---

## What is PrismLib?

PrismLib is **not** another vector database or Redis wrapper. It is a **tensor-native data plane** that sits next to your app and removes three expensive bottlenecks:

| Bottleneck | PrismLib answer | Component |
|------------|-----------------|-----------|
| Repeated LLM calls for similar questions | Semantic cache — paraphrases hit in-process | **PrismCache** |
| Every read goes to the DB over the network | WAL stream → local vector index on the app node | **PrismDriver** |
| Every agent re-embeds retrieval results | Provider embeds once; consumer gets float32 vectors | **PrismAPI** |
| Multi-container agents duplicate work | Shared answers + health mesh + failover | **PrismLib Micro** |

Everything runs **in-process** on your machines. Optional extras add the DB-node daemon (`prism-wrapper`), enterprise HTTP (`[enterprise]`), and Helm/Docker deploy templates. No mandatory Redis, Pinecone, or Kubernetes operator.

### The four layers (install what you need)

| Layer | One-line description | Azure-validated headline | Extra |
|-------|---------------------|--------------------------|-------|
| **PrismCache** | Wrap any LLM call; similar queries return cached answers | **91–96% hit rate** under load | `[cache]` |
| **PrismDriver** | Subscribe to DB WAL; query a local index instead of the network | **439× faster reads** (118ms → 0.27ms) | `[fabric]` |
| **PrismAPI** | HTTP/CHORUS API — pre-projected vectors for agents & RAG | 83% fewer consumer embed calls (design) | `[enterprise]` |
| **PrismLib Micro** | Cluster-wide token cache, alerts, Blue/Green/Orange failover | **76% fewer tokens** cluster-wide | `[fabric]` |

Deep dive: [RESULTS_AND_IMPROVEMENTS.md](RESULTS_AND_IMPROVEMENTS.md) · [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) · [ENTERPRISE.md](ENTERPRISE.md)

### How it fits together

```
Your app
  ├── PrismCache      → skip LLM when the question is semantically similar
  ├── PrismDriver     → skip DB network when the row vector is already local
  ├── PrismAPI client → skip re-embedding retrieval results from a provider
  └── ClusterCache    → skip LLM when another node already answered

DB node (optional)
  └── prism-wrapper   → reads WAL/binlog, streams encrypted vectors via CHORUS gRPC
```

Built on **[PrismResonance](https://github.com/insightitsGit/prismresonance)** (wave-memory similarity) and **[CHORUS Fabric](https://github.com/insightitsGit/chorus_fabric)** (encrypted float32 streaming).

---

### PrismAPI — vector-native API for agents (new in 0.7.0)

The provider embeds and projects document vectors **once**. Consumers receive CHORUS `API_RESPONSE` frames with float32 vectors — no second embedding pass on retrieval.

```python
from prism.api import PrismAPIProvider, PrismAPIClient, AuthConfig
from prism.lib.lang import PrismProjector, ProjectionConfig

provider = PrismAPIProvider(projector=..., embedder=..., semantic_fields=["title", "body"])

@provider.expose
def search(query: str, top_k: int = 10) -> list[dict]:
    return my_db.search(query)[:top_k]

# Consumer (agent / LangGraph node)
client = PrismAPIClient(projector, embedder, host="api.example.com", port=9100, api_key="...")
response = client.query("shipping policy", top_k=5)
stacked_vectors = response.vectors  # ready for PrismResonance / reranker
```

Enterprise HTTP server with auth + metrics:

```bash
pip install "prismlib-plus[enterprise,cache,fabric]"
python examples/enterprise_server.py
```

See [ENTERPRISE.md](ENTERPRISE.md) and [RELEASE_NOTES.md](RELEASE_NOTES.md).

### PrismCache — single node, in-process LLM cache

Wraps any LLM call. Paraphrased queries return the cached answer without touching the API.
Multi-tenant math: JL projection seeded by `SHA-256(tenant_id)` gives each tenant a mathematically
isolated address space — not a query filter, a projection matrix.

### PrismDriver — two-node WAL-streaming DB driver

Two components on two machines:
- **Server Wrapper** (DB node) — intercepts WAL/binlog, vectorizes rows, streams encrypted float32 frames via CHORUS Fabric
- **DLL Driver** (app node) — subscribes to the stream, keeps a local PrismResonance index warm; reads never leave the process

### PrismLib Micro — cluster cache, health mesh, Blue/Green/Orange failover

Built into `prismlib-plus[fabric]`, zero extra install:
- **ClusterCache** — once any node answers a query, every peer caches it via CHORUS TOKEN_SYNC frames. BLUE and ORANGE nodes billed 0 tokens on warm queries.
- **AlertManager** — 12 default health rules; fires SIGNAL frame + admin email in <1s when CPU/RAM/disk thresholds are crossed. No scrape interval. No Datadog agent.
- **Blue/Green/Orange failover** — GREEN is active master, BLUE is warm standby (auto-promotes in ~3s if GREEN goes silent), ORANGE is syncing reserve.
- **ContextCompressor** — cosine-sim top-K chunk selection before every LLM call. 58–64% context token reduction, zero extra cost.

---

Built on two open-source InsightIts libraries:
- **[PrismResonance](https://github.com/insightitsGit/prismresonance)** — wave-memory similarity engine powering every cache lookup and local vector index
- **[CHORUS Fabric](https://github.com/insightitsGit/chorus_fabric)** — encrypted gRPC binary streaming protocol carrying float32 tensor frames between nodes

---

## Installation

```bash
# Semantic LLM cache
pip install "prismlib-plus[cache]"

# DB driver (app node)
pip install "prismlib-plus[fabric]"

# Enterprise PrismAPI (auth, metrics, FastAPI)
pip install "prismlib-plus[enterprise,cache,fabric]"

# Server Wrapper daemon (DB node — Linux/macOS)
pip install "prismlib-plus[wrapper]"
prism-wrapper --grpc-port 50051

# Everything
pip install "prismlib-plus[all,enterprise]"
```

**Legacy base package (PyPI, no PrismAPI):** `pip install "prismlib[cache]"` — [prismlib 0.4.0](https://pypi.org/project/prismlib/).

<details>
<summary>Previous install snippets (prismlib 0.4.0 naming)</summary>

```bash
pip install "prismlib[cache]"
pip install "prismlib[fabric]"
pip install "prismlib[all]"
```
</details>

---

## Use Cases

### PrismCache

#### Drop-in LLM response cache

Save 60-80% of LLM API calls by serving semantically identical queries from cache.
Paraphrases hit the cache — "How do I reset my password?" and "I forgot my password, help" return the same answer without a second LLM call.

```python
from prism.cache import PrismCache

cache = PrismCache.build(tenant_id="my-app", llm_model="gpt-4o")

def ask(question: str) -> str:
    return cache.get_or_call(
        query=question,
        call_fn=lambda: openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
        ).choices[0].message.content,
    )
```

#### Multi-tenant SaaS — isolated caches per customer

Each tenant gets a mathematically isolated cache space (JL projection seeded by tenant ID).
One customer's cached answers never bleed into another's.

```python
from prism.cache import PrismCache

def get_cache(tenant_id: str) -> PrismCache:
    return PrismCache.build(tenant_id=tenant_id, llm_model="gpt-4o-mini")

# Tenant A and tenant B share no cache state
cache_a = get_cache("acme-corp")
cache_b = get_cache("globex-inc")

answer = cache_a.get_or_call(query="What is my plan limit?", call_fn=llm_call)
```

#### FastAPI / Django middleware — transparent caching

Wrap your existing LLM endpoint without changing any business logic.

```python
# FastAPI
from fastapi import FastAPI, Request
from prism.cache import PrismCache

app = FastAPI()
cache = PrismCache.build(tenant_id="api", llm_model="gpt-4o")

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    question = body["message"]
    answer = await cache.aget_or_call(
        query=question,
        call_fn=lambda: llm_client.ask(question),
    )
    return {"answer": answer}
```

```python
# Django — add to MIDDLEWARE in settings.py
# prism/middleware.py
from prism.cache import PrismCache

_cache = PrismCache.build(tenant_id="django-app", llm_model="gpt-4o")

class PrismCacheMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_llm_query(self, question: str, call_fn) -> str:
        return _cache.get_or_call(query=question, call_fn=call_fn)
```

#### Async batch queries

```python
import asyncio
from prism.cache import PrismCache

cache = PrismCache.build(tenant_id="batch", llm_model="gpt-4o-mini")

async def process_batch(questions: list[str]) -> list[str]:
    tasks = [
        cache.aget_or_call(query=q, call_fn=lambda q=q: llm_call(q))
        for q in questions
    ]
    return await asyncio.gather(*tasks)
```

#### Cost estimation

```python
from prism.cache import PrismCache

cache = PrismCache.build(tenant_id="finance", llm_model="gpt-4o")

# After processing queries...
metrics = cache.metrics()
print(f"Hit rate:          {metrics.hit_rate:.0%}")
print(f"Tokens saved:      {metrics.tokens_saved:,}")
print(f"Cost saved today:  ${metrics.cost_saved_usd:.2f}")
print(f"Projected monthly: ${metrics.cost_saved_usd * 30:.0f}")
```

---

### PrismDriver

PrismDriver has two components that work together. Install each on the right machine.

**On the DB node — Server Wrapper**

The Server Wrapper is an OS daemon that sits next to your database. It reads WAL/binlog changes, vectorizes rows using `RowVectorizer`, encrypts them with `TensorCipher` (via CHORUS Fabric), and streams float32 frames to every connected DLL Driver.

```bash
# Install on the DB node (Linux or macOS)
pip install "prismlib[wrapper]"

# Configure and start
prism-wrapper --config /etc/prism/wrapper.toml
```

```toml
# /etc/prism/wrapper.toml
[database]
flavor = "postgresql"
dsn = "postgresql://user:pass@localhost/mydb"

[chorus]
listen_port = 50051
tenant_id = "products-service"
```

Supported databases: PostgreSQL (WAL / wal2json), MySQL (binlog), CockroachDB (EXPERIMENTAL CHANGEFEED), TiDB (push model).

**On the app node — DLL Driver**

The DLL Driver is an in-process library that replaces your DB connection string. On startup it connects to the Server Wrapper, subscribes to the CHORUS Fabric stream, and keeps a local PrismResonance index warm. All reads hit the in-process index — no network round-trip, sub-millisecond latency.

```bash
# Install on the app node
pip install "prismlib[fabric]"
```

#### Replace your DB connection string

```python
# Before
import psycopg2
conn = psycopg2.connect("postgresql://user:secret@db-host:5432/mydb")

# After — no password, no hostname in app config
from prism.ffi import PrismDriver, DriverConfig

async with PrismDriver(DriverConfig(wrapper_host="db-proxy-1")) as driver:
    results = await driver.query(
        embedding=my_embedding_vector,
        top_k=5,
        threshold=0.85,
    )
```

#### Sub-millisecond row lookups via local cache

The driver keeps a local PrismResonance cache warm via a background WAL subscription.
Reads never touch the DB — they hit the in-process float32 index.

```python
from prism.ffi import PrismDriver, DriverConfig
import numpy as np

config = DriverConfig(
    wrapper_host="10.0.1.50",
    wrapper_port=50051,
    tenant_id="products-service",
)

async with PrismDriver(config) as driver:
    # Typical hit: < 1ms, no network round-trip
    query_vec = np.array([...], dtype=np.float32)
    matches = await driver.query(embedding=query_vec, top_k=10)
    for m in matches:
        print(f"{m.row_id}  score={m.score:.3f}  {m.text_repr}")
```

#### Write through to DB

```python
async with PrismDriver(config) as driver:
    ack = await driver.write(
        row_id="product-42",
        data={"name": "Widget Pro", "price": 29.99, "stock": 150},
    )
    print(f"Written: event_id={ack.event_id}")
```

#### Go, C#, PHP, Java — same DLL, native bindings

```go
// Go
import prism "github.com/insightitsGit/prismlib/go"

driver, _ := prism.Connect("db-proxy-1:50051", "my-tenant")
defer driver.Close()
results, _ := driver.Query(embedding, prism.QueryOpts{TopK: 5, Threshold: 0.85})
```

```csharp
// C#
using InsightIts.Prism;

await using var driver = new PrismDriver("db-proxy-1:50051", tenantId: "my-tenant");
await driver.ConnectAsync();
var results = await driver.QueryAsync(embedding, topK: 5, threshold: 0.85f);
```

```php
// PHP 8.0+
$driver = new PrismDriver('db-proxy-1', 50051, 'my-tenant');
$driver->connect();
$results = $driver->query($embedding, topK: 5, threshold: 0.85);
```

---

## Architecture

```
┌─ DB Node ──────────────────────────────────────────────────────┐
│  PostgreSQL / MySQL / CockroachDB / TiDB                       │
│       │ WAL / binlog / changefeed                              │
│  ┌────▼───────────────────────────────────────────────────┐    │
│  │  prism-wrapper  (pip install "prismlib[wrapper]")      │    │
│  │  RowVectorizer → TensorCipher (V_enc = V @ K)         │    │
│  │  → HMAC-SHA256 watermark → CHORUSPublisher            │    │
│  └────────────────────────┬───────────────────────────────┘    │
└───────────────────────────┼────────────────────────────────────┘
                            │  CHORUS Fabric (gRPC, encrypted float32)
┌─ App Node — GREEN ────────┼────────────────────────────────────┐
│  ┌────────────────────────▼──────────────────────────────┐     │
│  │  PrismDriver DLL  (pip install "prismlib[fabric]")    │     │
│  │  Subscribe loop → decrypt → PrismResonance index      │     │
│  └──────────────────────────┬────────────────────────────┘     │
│                             │ sub-ms query                     │
│  ┌──────────────────────────▼────────────────────────────┐     │
│  │  Your Application                                      │     │
│  │  ┌─────────────────┐   ┌──────────────────────────┐   │     │
│  │  │  PrismCache     │   │  PrismDriver             │   │     │
│  │  │  LLM cache      │   │  local PrismResonance    │   │     │
│  │  │  [cache]        │   │  (no DB round-trip)      │   │     │
│  │  └─────────────────┘   └──────────────────────────┘   │     │
│  │  ┌──────────────────────────────────────────────────┐  │     │
│  │  │  ClusterCache  ← TOKEN_SYNC frames               │  │     │
│  │  │  AlertManager  ← HEALTH / SIGNAL frames          │  │     │
│  │  └──────────────────────────────────────────────────┘  │     │
│  └────────────────────────────────────────────────────────┘     │
└──────────────────────────────┬─────────────────────────────────┘
                               │  CHORUS mesh
          ┌────────────────────┴────────────────────┐
          │  TOKEN_SYNC · HEALTH · SIGNAL · CONFIG   │
          ▼                                          ▼
┌─ App Node — BLUE ──────┐           ┌─ App Node — ORANGE ─────┐
│  ClusterCache          │           │  ClusterCache            │
│  (warm standby)        │           │  (syncing reserve)       │
│  auto-promotes if      │           │  separate network        │
│  GREEN silent >3s      │           │                          │
└────────────────────────┘           └──────────────────────────┘
```

---

## Benchmark

### PrismCache — semantic LLM cache

Live results from Azure Container App (`westus2`, 1 vCPU / 2 GiB, mock LLM baseline):

| Scenario | Users | Duration | Hit rate | Queries | Tokens saved | Monthly est. |
|----------|-------|----------|----------|---------|-------------|-------------|
| Light    | 20    | 60s      | **91.0%** | 5,936  | 1,374,464   | **$594**    |
| Mixed    | 50    | 300s     | **95.9%** | 6,973  | 1,673,216   | **$723**    |

> Numbers use a mock LLM (80ms sleep). With real GPT-4o calls (1–3s), latency speedup is 4–13×; token savings are identical.

### PrismDriver — two-node baseline vs local index (Azure e2e, 2026-06-29)

Latest two-container benchmark (`deploy/azure_driver_run.ps1`, westus2):

| Phase | Path | Avg latency | Queries |
|-------|------|-------------|---------|
| **Baseline** | App → DB node over network | **118.5 ms** | 2,653 |
| **Driver** | App → in-process index | **0.27 ms** | 4,886 |

**439× faster · 99.8% latency reduction** (20 users × 45 s/phase, Python driver mode)

Previous run (2026-06-24): 70.7× / 142.8 ms → 2.0 ms — see `driver_benchmark_20260624_135338.json`.

```bash
python benchmark/load/run_driver_benchmark.py \
  --app-url https://prism-benchmark.<your-env>.azurecontainerapps.io \
  --db-url  https://prism-wrapper-sim.<your-env>.azurecontainerapps.io \
  --users 20 --duration 45
```

See [`benchmark/`](benchmark/) for full results JSON, Locust CSV files, and the Azure deploy script.

---

## Core libraries

PrismLib is built on two InsightIts open-source libraries. You can use them directly if you need lower-level access.

### PrismResonance

> **[github.com/insightitsGit/prismresonance](https://github.com/insightitsGit/prismresonance)** · `pip install prismresonance`

The wave-memory similarity engine. Every cache lookup and local vector index in PrismLib goes through PrismResonance.

How it works:
- Receives a float32 embedding vector
- Johnson-Lindenstrauss reduces it to 64 dimensions using a projection matrix seeded by `SHA-256(tenant_id)` — this is what gives each tenant mathematically isolated address space
- Computes similarity as wave interference (cosine in projected space) in three lock-free phases: snapshot → ONNX MatMul → rank
- Returns ranked candidates in sub-millisecond time entirely in-process

PrismCache wraps this for LLM response caching. PrismDriver's local replica is a PrismResonance index kept warm by WAL streaming.

```python
from prismresonance import PrismProjector, WaveIndex

projector = PrismProjector(dim=64, tenant_id="my-tenant")
index = WaveIndex(projector)

index.add(vector=my_embedding, payload={"row_id": "product-1", "text": "Widget"})
results = index.query(vector=query_embedding, top_k=5, threshold=0.85)
```

### CHORUS Fabric

> **[github.com/insightitsGit/chorus_fabric](https://github.com/insightitsGit/chorus_fabric)** · `pip install chorus-fabric`

The secure gRPC binary streaming protocol for machine-to-machine tensor communication. PrismDriver uses CHORUS Fabric as its transport layer between the server wrapper on the DB node and the DLL driver on the app node.

How it works:
- `prism-wrapper` (DB node) vectorizes WAL row events via `RowVectorizer`, encrypts them with `TensorCipher` (`V_enc = V @ K`), appends an HMAC-SHA256 watermark, and publishes batches of raw float32 frames
- `PrismDriver` (app node) opens a persistent `WrapperService.Subscribe()` gRPC stream, receives encrypted frames, decrypts, and feeds them into the local PrismResonance index
- Transport is pure binary float32 over gRPC server-streaming — no JSON serialization, no REST overhead
- The `WrapperService` proto also exposes `Query`, `Write`, `Health`, and `Hello` RPCs for direct interaction

```python
from chorus_fabric import CHORUSPublisher, DriverEndpoint

publisher = CHORUSPublisher(config)
publisher.add_driver(DriverEndpoint(host="10.0.1.50", port=50051, tenant_id="prod"))
await publisher.run(event_queue)  # streams WAL events to all connected drivers
```

CHORUS Fabric is the same protocol used in the CHORUS M2M system — InsightIts' 4-container gRPC topology for tensor communication between AI agents. The 98.6% latency reduction in the PrismDriver benchmark is direct proof that the protocol works at production scale: 11,000 rows streamed at 26,000 rows/s across Azure inter-container networking, then served locally at 2ms.

---

## PrismLib Micro — Cluster & RAG Layer

PrismLib Micro is the cluster layer built into `prismlib[fabric]`. It adds three
capabilities on top of the single-node stack — no extra install, no extra infra.

### What's included

| Component | What it does |
|-----------|-------------|
| **ClusterCache** | Shares LLM answers across all nodes via CHORUS TOKEN_SYNC frames. Once any node answers a query, every other node serves it for 0 tokens. |
| **AlertManager** | Broadcasts health alerts as SIGNAL frames + admin email the moment CPU/RAM/disk/latency thresholds are crossed. No Prometheus. No Datadog. |
| **Blue/Green/Orange failover** | Three-tier hot-standby: GREEN (active), BLUE (warm standby, auto-promotes in ~3s), ORANGE (syncing reserve). No Raft dependency. No K8s operator. |
| **ContextCompressor** | Ranks RAG context chunks by cosine similarity, keeps top-K. Saves 58–64% of context tokens before every LLM call. In-process, no extra model. |

### Cluster benchmark results (3-node, Azure Container Apps · 2 VNets · westus2)

| Metric | Result |
|--------|--------|
| Token savings — cluster avg | **76.1%** |
| BLUE node (cluster cache hit) | **100%** — 0 LLM calls |
| ORANGE node (cross-VNet cache hit) | **100%** — 0 LLM calls |
| Context compression | **58–64%** per query |
| CHORUS frame latency (cross-VNet) | **~22 ms** (same-region) |
| Health alert propagation | **633–674 ms** measured |
| Failover — BLUE promoted to GREEN | **~4 s** detect + **97 ms** promote |

See [`benchmark/cluster/`](benchmark/cluster/) for the benchmark code, [`deploy/azure_cluster_run.sh`](deploy/azure_cluster_run.sh) for the Azure deploy, and [`benchmark/cluster/cluster_benchmark_results_azure.json`](benchmark/cluster/cluster_benchmark_results_azure.json) for raw results.

**Full results in one place:** [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) — every cache, driver, and cluster number with sources. **The design & novelty:** [`whitepaper_chorus_mesh.md`](whitepaper_chorus_mesh.md) (cache-replication traffic doubling as the failure detector).

### ClusterCache — 5-line RAG integration

```python
from prism.cluster.cache import ClusterCache

cache = ClusterCache(node_id="node-1", fabric=chorus_fabric)

answer = await cache.get_or_call(
    query          = user_question,
    query_vector   = embed(user_question),
    call_fn        = lambda: llm.complete(user_question),
    context_chunks = retrieved_docs,    # your RAG chunks
    chunk_vectors  = doc_embeddings,    # their vectors
)
```

Drop this in front of your existing `retrieve → generate` step. No changes to
retrieval logic, no changes to your LLM client.

### AlertManager — email + SIGNAL frame on health threshold

```python
from prism.cluster.alerts import AlertManager, SMTPConfig

alerts = AlertManager(
    fabric = chorus_fabric,
    mail_config = SMTPConfig(
        host="smtp.gmail.com", port=587,
        username="you@gmail.com",
        password=os.getenv("GMAIL_APP_PASS"),
        recipients=["admin@yourcompany.com"],
    ),
)
await alerts.evaluate_health(health_snapshot)
# Fires email + SIGNAL frame to all nodes if any of the 12 default rules trigger
```

### Competitive position

| Capability | PrismLib Micro | Prometheus + Alertmanager | Redis cluster | Raft / etcd |
|-----------|---------------|--------------------------|---------------|-------------|
| Cross-node token cache | **Yes, built-in** | No | Manual (exact match) | No |
| Alert propagation | **<1 s, no infra** | 30–60 s, stack needed | No | No |
| Auto failover | **~3–4 s, built-in** | No | Sentinel, 2–30 s | **150–500 ms** |
| Context compression | **58–64%, free** | No | No | No |
| Extra infrastructure | **None** | Prometheus stack | Redis cluster | etcd cluster |

### Pricing

| Tier | Nodes | Price | Includes |
|------|-------|-------|---------|
| **Open source** | Unlimited | **Free forever** | All cluster code, Apache 2.0 |
| **ChorusMesh Developer** *(coming soon)* | Up to 3 | $29/mo after 30-day trial | ClusterCache + failover + AlertManager |
| **ChorusMesh Team** | Up to 10 | $149/mo | + Raft consensus, message broker adapters |
| **ChorusMesh Business** | Up to 50 | $499/mo | + multi-region routing, SLA 99.9% |
| **Enterprise** | Unlimited | Contact us | + air-gap, compliance, dedicated Slack |

For enterprise agreements: **[insightits.info@gmail.com](mailto:insightits.info@gmail.com)**

---

## Enterprise (0.7.0)

Open-source core (Apache 2.0) plus an **optional enterprise layer**:

| Feature | Module / doc |
|---------|----------------|
| PrismAPI provider & consumer | `prism.api` |
| API keys, rate limits, audit | `prism.api.auth`, `prism.security` |
| Prometheus + optional OpenTelemetry | `prism.observability` |
| mTLS (wrapper gRPC + driver) | `prism.security.tls`, `SECURITY.md` |
| MCP tool server | `prism.api.mcp` |
| Helm + Docker | `deploy/helm/`, `deploy/Dockerfile.enterprise` |
| Runnable server | `examples/enterprise_server.py` |

**Deploy guide:** [ENTERPRISE.md](ENTERPRISE.md) · **Release notes:** [RELEASE_NOTES.md](RELEASE_NOTES.md)

For SLA-backed support, air-gapped installs, and multi-region topologies: **[insightits.info@gmail.com](mailto:insightits.info@gmail.com)**

---

## Sponsors

PrismLib is free and will stay free. If it saved your team money on OpenAI bills or database infrastructure, consider sponsoring — it covers benchmark compute, maintenance time, and keeps development moving.

[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?logo=github)](https://github.com/sponsors/insightitsGit)

<!-- sponsors -->
*Your name or logo here — [become a sponsor](https://github.com/sponsors/insightitsGit)*
<!-- /sponsors -->

---

## Publishing to PyPI

This repo publishes as **`prismlib-plus`** (superset of PyPI [`prismlib` 0.4.0](https://pypi.org/project/prismlib/)).

```bash
pip install "prismlib-plus[cache]"           # PrismCache
pip install "prismlib-plus[fabric]"          # PrismDriver + cluster
pip install "prismlib-plus[wrapper]"         # DB-node daemon
pip install "prismlib-plus[enterprise]"      # PrismAPI + auth + metrics
pip install "prismlib-plus[all,enterprise]"  # Everything
```

**To publish 0.7.0:** see [RELEASE.md](RELEASE.md).  
**What's new:** [RELEASE_NOTES.md](RELEASE_NOTES.md) · [CHANGELOG.md](CHANGELOG.md)

---

## License

Apache 2.0 — InsightIts © 2026

---

## Links

- Author: **Amin Parva** ([insightits.info@gmail.com](mailto:insightits.info@gmail.com))
- Company: [https://www.insightits.com](https://www.insightits.com)
- GitHub: https://github.com/insightitsGit/prismlibplusapi
- PyPI: https://pypi.org/project/prismlib-plus/
- Company site: https://www.insightits.com
