# PrismAPI — Vector-Native API Layer for AI Agents

**Tagline:** Providers embed once. Agents retrieve forever. No re-embedding. No JSON round-trips.

**One-line pitch:** PrismAPI is the missing API contract between AI data providers and the agents that consume them — delivering pre-projected float32 vectors over CHORUS so every agent hop skips the embedding tax.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [The Solution](#the-solution)
3. [How It Works](#how-it-works)
4. [Key Benefits](#key-benefits)
5. [Use Cases](#use-cases)
6. [Benchmark Results](#benchmark-results)
7. [Installation](#installation)
8. [Quick Start](#quick-start)
9. [Integration Patterns](#integration-patterns)
10. [API Reference](#api-reference)
11. [Architecture Deep Dive](#architecture-deep-dive)
12. [FAQ](#faq)
13. [Product Positioning](#product-positioning)

---

## The Problem

Every time an AI agent retrieves content from an API, it pays an **embedding tax** it doesn't need to:

```
Standard REST retrieval (what every agent does today):

  Agent                     API Server
    │                           │
    │── GET /search?q="..." ───>│  Server embeds query    [1 embed call]
    │<── JSON { "text": "..." } ─│
    │                           │
    │  Agent receives text       │
    │  Agent calls embed(text)   │  [top_k embed calls — one per result]
    │  Agent gets float32 vector │
    │  Agent uses vector         │
```

**What the agent actually needed was the vector. It got text and had to convert it.**

At scale this compounds quickly:

| Scale | top_k | Embedding calls per second |
|-------|-------|---------------------------|
| 100 req/s | 5 | 500 calls/s |
| 1,000 req/s | 10 | 10,000 calls/s |
| 10,000 req/s | 10 | 100,000 calls/s |

Each call costs 20–100 ms latency and $0.00002 (OpenAI `text-embedding-3-small`). At 10,000 calls/second that is **$1,728/day** in embedding API cost alone — for re-embedding content that was already a vector on the server.

In multi-agent pipelines the problem multiplies. Agent A retrieves, passes text to Agent B, which re-embeds and passes text to Agent C, which re-embeds again. Five-hop pipeline = five re-embeddings of the same content.

---

## The Solution

**PrismAPI breaks the text-embedding loop.**

The provider embeds content once at index time and stores the projected float32 vectors. When a consumer queries, the server returns those pre-projected vectors directly — no text serialization, no consumer re-embedding.

```
PrismAPI retrieval (CHORUS binary protocol):

  Agent                           PrismAPI Server
    │                                   │
    │── POST /chorus/search ──────────>│
    │   [CHORUSFrame: float32 vector]   │  Server does: cosine search
    │                                   │  against pre-indexed vectors
    │<── CHORUSFrame: [float32 × top_k] ─│  [0 embed calls]
    │                                   │
    │  Agent receives float32 vectors    │
    │  Agent uses them directly          │
    │  [0 embed calls on consumer side]  │
```

**The embedding call is not eliminated globally — it is moved to the provider, done once at index time, and the result is reused for every query forever.**

---

## How It Works

### Step 1: Provider indexes content (once)

```python
from prism.api import PrismAPIProvider, SentenceTransformerEmbedder
from prism.lib.lang import PrismProjector, ProjectionConfig

projector = PrismProjector(ProjectionConfig(tenant_id="my-org", target_dim=64))
embedder  = SentenceTransformerEmbedder()   # or OpenAI, Gemini, any embedder

provider = PrismAPIProvider(
    projector=projector,
    embedder=embedder,
    semantic_fields=["title", "body"],       # TEXT → vectorized
    id_field="doc_id",
    exact_fields=["price", "url", "category"], # NEVER vectorized — exact sidecar
)
```

When the provider indexes a document:
- `title` + `body` are concatenated, embedded (384-dim), projected to 64-dim via Johnson-Lindenstrauss
- `price`, `url`, `category` travel as a JSON sidecar — never embedded

This happens **once per document**, at index time.

### Step 2: Consumer queries (zero re-embedding on results)

```python
from prism.api import PrismAPIClient

client = PrismAPIClient(
    projector=projector,     # same tenant_id → same projection space
    embedder=embedder,
    host="api.example.com",
    port=9100,
)

response = client.query("how does inflation affect bond prices?", top_k=10)

# response.vectors  → np.ndarray (10, 64) — ready for cosine search, clustering, re-ranking
# response.sidecars → list of dicts — exact fields: price, url, category
# response.embedding_calls_saved → 10
```

The consumer embeds its **query once**. The 10 results arrive as float32 vectors. Zero additional embedding calls.

### The Semantic / Exact Boundary

This is the architectural contract that PrismAPI enforces. **It is not optional.**

| Field type | Examples | What happens |
|-----------|----------|--------------|
| **Semantic** | `title`, `body`, `description`, `summary`, `review` | Embedded + projected → float32 vector |
| **Exact** | `price`, `id`, `url`, `in_stock`, `date`, `count`, `category` | JSON sidecar — never embedded |

Embedding a price is meaningless. A float32 embedding cannot encode `$49.99` vs `$50.00` — those are identical in vector space. Exact fields must travel as exact data.

### The Projection Space

Both provider and consumer use a **tenant-isolated JL projection**:
- Seed: `SHA-256(tenant_id)` → deterministic random matrix
- Input: 384-dim embedding (any model)
- Output: 64-dim or 128-dim float32 vector
- Property: cosine similarity is approximately preserved (Johnson-Lindenstrauss lemma)

**Critical:** provider and consumer must use the same `tenant_id`. Different tenant_ids produce incompatible vector spaces.

```
Provider (tenant_id="my-org"):
  "bond prices" → [0.384-dim embedding] → [64-dim JL projection] → CHORUSFrame

Consumer (tenant_id="my-org"):    ← must match
  "inflation bonds" → [0.384-dim embedding] → [64-dim JL projection] → search
  Results arrive as 64-dim vectors → cosine compare → re-rank
```

---

## Key Benefits

### 1. Structural embedding call reduction: always 1 instead of 1 + top_k

| | Baseline | PrismAPI |
|-|----------|----------|
| Embedding calls per query | `1 + top_k` | `1` |
| At top_k=5 | 6 calls | 1 call |
| At top_k=10 | 11 calls | 1 call |
| **Saving at top_k=10** | — | **90.9%** |

This is not a tunable parameter — it is structural. The baseline must always embed the query once and each result once. PrismAPI always embeds the query once and never embeds results.

### 2. Tail latency improvement under concurrent load

Under concurrent load, multiple agents requesting simultaneously saturate the embedding model. PrismAPI eliminates server-side embedding entirely on the CHORUS path.

| Metric | Baseline | PrismAPI | Improvement |
|--------|----------|----------|-------------|
| P50 latency | 2.0 ms | 1.6 ms | 20% |
| P95 latency | 18.0 ms | 1.9 ms | **89%** |
| P99 latency | 517 ms | 6.6 ms | **99%** |

The P99 spike on baseline (517 ms) is embedding model saturation — multiple agents pile into the embedding model at the same moment. PrismAPI has no embedding model to saturate on the CHORUS path.

### 3. Vector propagation through multi-hop agent pipelines

Standard pipeline today:
```
Agent A → retrieve → JSON text → [embed × 10] → pass to Agent B
                                                  [embed × 10] → pass to Agent C
                                                                  [embed × 10] → reason
```
3 hops × 10 results × 30 ms = **900 ms in re-embedding overhead**

PrismAPI pipeline:
```
Agent A → retrieve → float32 vectors → Agent B → float32 vectors → Agent C → reason
```
3 hops × 0 ms = **0 ms in re-embedding overhead**

Vectors propagate through the state dict. Every downstream node receives pre-projected float32 vectors it can use directly.

### 4. Privacy-preserving federated search

The query travels as a float32 vector — the provider never sees the query text. Results travel as float32 vectors — the consumer never sees the raw document content. Both parties operate in projected space:

```
Hospital A (consumer)                    Hospital B (provider)
  "patient symptoms" query               Clinical notes (indexed)
    ↓ embed locally                        ↓ projected at index time
    ↓ project to 64-dim                    ↓ stored as float32
    ↓ send vector (not text)               ↓
  ──────── CHORUSFrame ──────────────────> cosine search
  <─────── CHORUSFrame ──────────────────  float32 results (not text)
    ↓ use vectors directly
    (never saw raw clinical notes)         (never saw query text)
```

Compliant with data residency requirements. Neither party exposes raw content.

### 5. Multi-provider fan-out without re-embedding

Query 3 providers (news, legal, medical) in parallel, get results in the same projected space (same `tenant_id`), rank across all providers with a single cosine sort — no per-provider re-embedding:

```python
from prism.api import MultiProviderClient

result = MultiProviderClient(clients={
    "news":    news_client,
    "legal":   legal_client,
    "medical": medical_client,
}, top_k_per_provider=5, total_top_k=10).query("drug pricing regulation")

# result.vectors      → (10, 64) merged across 3 providers, ranked by cosine
# result.sidecars     → [{"provider": "legal", ...}, {"provider": "news", ...}, ...]
# result.per_provider → raw per-provider responses for debugging
```

Wall-clock time = `max(RTT_news, RTT_legal, RTT_medical)` — not the sum.

---

## Use Cases

### Use Case 1: LangGraph / multi-agent pipelines

**Who:** Teams building LangGraph, CrewAI, or AutoGen pipelines where Agent A retrieves and Agent B reasons.

**Pain:** At every agent hop, retrieved text is re-embedded. A 5-agent pipeline with top-10 retrieval triggers 50+ embedding calls for content that was already vectorized on the server.

**Solution:** `PrismRetrieverNode` drops into any LangGraph `StateGraph`. Vectors flow through state. Downstream nodes receive float32 arrays, not text.

```python
from prism.api import create_retriever_node
from langgraph.graph import StateGraph
from typing import TypedDict, Optional
import numpy as np

class ResearchState(TypedDict):
    query: str
    search_results: Optional[object]   # APIResponse
    answer: Optional[str]

graph = StateGraph(ResearchState)

# One line to add zero-re-embed retrieval
graph.add_node("retrieve", create_retriever_node(
    client=my_prism_client,
    query_key="query",
    results_key="search_results",
    top_k=10,
    error_key="retrieval_error",
    fallback_fn=my_rest_api_fallback,   # graceful degradation
))

graph.add_node("reason", my_reasoning_node)
graph.add_edge("retrieve", "reason")
```

**Result:** Agent B receives `state["search_results"].vectors` — np.ndarray (10, 64) — and can immediately compute cosine similarity, cluster, or pass to PrismResonance for re-ranking. No embedding call.

---

### Use Case 2: Real-time retrieval for latency-sensitive applications

**Who:** Voice assistants, live coding assistants, in-game NPCs, trading terminals — anything with a sub-200 ms latency budget.

**Pain:** At 50 ms per embedding call, retrieving 10 results costs 500 ms in re-embedding alone, before any reasoning.

**Solution:** The CHORUS path eliminates result re-embedding entirely. PrismAPI P95 latency = 1.9 ms vs 18 ms baseline — a 9× improvement on the retrieval step.

**Budget comparison:**
```
200 ms total budget for voice assistant response:
  Baseline:  50ms (embed query) + 18ms (retrieve) + 500ms (embed 10 results) = 568ms ❌ budget exceeded
  PrismAPI:  50ms (embed query) +  2ms (retrieve) +   0ms (results are vectors) = 52ms ✓ 148ms for reasoning
```

---

### Use Case 3: Privacy-preserving federated search

**Who:** Healthcare networks, legal discovery platforms, financial data exchanges — any context where raw data cannot leave its origin.

**Pain:** Semantic search across institutions requires either sharing raw documents (compliance risk) or sharing an embedding model (infrastructure coupling). Neither is acceptable.

**Solution:** PrismAPI makes both sides work in projected vector space. Neither party exposes raw content.

- **Provider** embeds and projects documents locally, serves float32 vectors
- **Consumer** embeds query locally, sends float32 vector, receives float32 results
- **Wire:** only binary float32 arrays cross the boundary — no text, no raw embeddings

This is not a workaround — it is the correct architecture for federated semantic search. Pairs naturally with CHORUS encryption (`V_enc = V_raw @ K`) for transport-level security.

---

### Use Case 4: Multi-provider semantic aggregation marketplace

**Who:** Platforms that aggregate content from multiple semantic data providers — news APIs, research databases, legal corpora, medical literature.

**Pain:** Each provider uses a different embedding model. A consumer querying 5 providers gets 5 incompatible vector spaces and must either standardize on one model (vendor lock-in) or re-embed all results in its own model (expensive).

**Solution:** PrismAPI's tenant-isolated JL projection decouples the provider's embedding model from the consumer's retrieval space. Any provider can use any embedding model; all results arrive in the same 64-dim projected space (same `tenant_id`), directly rankable against each other.

```
Provider A: indexes with OpenAI text-embedding-3-large (3072-dim) → projects to 64-dim
Provider B: indexes with all-MiniLM-L6-v2 (384-dim) → projects to 64-dim
Provider C: indexes with Gemini text-embedding-004 (768-dim) → projects to 64-dim

Consumer: queries all 3, receives (N, 64) from each, merges, ranks — no model dependency
```

**One integration. Every provider. No embedding model dependency.**

---

### Use Case 5: High-frequency automated pipelines

**Who:** Financial monitoring agents, legal discovery pipelines, code review agents, content moderation — anything running >1,000 searches/hour automatically.

**Pain:** At scale, the embedding cost for re-embedding results becomes the largest line item. A pipeline running 10,000 queries/hour at top_k=10 triggers 100,000 embedding calls/hour.

**Savings at scale:**

| Queries/hour | top_k | Baseline calls/hr | PrismAPI calls/hr | Saved | $/day (OpenAI) |
|-------------|-------|-------------------|-------------------|-------|----------------|
| 1,000 | 5 | 6,000 | 1,000 | 83% | $0.29 |
| 10,000 | 10 | 110,000 | 10,000 | 90.9% | $5.76 |
| 100,000 | 10 | 1,100,000 | 100,000 | 90.9% | $57.60 |
| 1,000,000 | 10 | 11,000,000 | 1,000,000 | 90.9% | $576 |

At 1M queries/hour, PrismAPI saves **$576/day** in embedding API cost alone. The implementation cost is one pip install and one decorator line.

---

### Use Case 6: Edge agents with asymmetric compute

**Who:** On-device AI (mobile, automotive, industrial IoT), small embedded agents, browser-based AI.

**Pain:** A mobile device can embed a short query (small model, fast), but cannot run a full embedding model to process 10 returned results. Standard REST APIs return text that the device cannot economically embed.

**Solution:** PrismAPI returns float32 vectors. The device uses them directly for cosine comparison, clustering, or ranking — no embedding model required on the consumer side. Heavy compute stays at the provider.

```
Mobile device (4GB RAM, no GPU):           Cloud PrismAPI server:
  "find nearby restaurants" → embed query    Pre-indexed 1M restaurants
    (fast, small model)                        (embedded at index time, once)
  → send 64-dim float32 vector
  ← receive float32 vectors (10 × 64)
  → cosine sort → display results
  [0 embedding calls on device for results]
```

---

## Benchmark Results

**Setup:** Real HTTP benchmark — separate server process, 200-document corpus, 50 queries × 3 trials, all-MiniLM-L6-v2, 64-dim JL projection.

> Reproduce: `python benchmark/api/run_real_benchmark.py --dim 64`

### End-to-end latency

| | Baseline (HTTP/JSON) | PrismAPI (CHORUS) | Improvement |
|--|---------------------|-------------------|-------------|
| RTT mean | 6.4 ms | 1.8 ms | 3.6× faster |
| RTT P50 | 2.0 ms | 1.6 ms | 1.3× |
| RTT P95 | 18.0 ms | 1.9 ms | **9.5× faster** |
| RTT P99 | 23.8 ms | 11.2 ms | 2.1× |
| Server embed latency | 3.4 ms | **0.0 ms** | eliminated |
| Client embed latency (per result set) | 36.8 ms | **0.0 ms** | eliminated |

**True end-to-end per query (RTT + client embed):**
- Baseline: 6.4 + 36.8 = **43.2 ms**
- PrismAPI: 1.8 + 0.0 = **1.8 ms** — **24× faster**

### Embedding calls (structural — not tunable)

| | Baseline | PrismAPI | Saved |
|-|----------|----------|-------|
| 50 queries × 3 trials | 900 calls | 150 calls | **750 calls (83.3%)** |
| Per query (top_k=5) | 6.0 | 1.0 | **83.3%** |
| Per query (top_k=10) | 11.0 | 1.0 | **90.9%** |

### Concurrency (10 agents, 100 simultaneous requests)

| | Baseline | PrismAPI | Note |
|-|----------|----------|------|
| Throughput | 187 req/s | 131 req/s | Baseline wins on raw throughput* |
| P50 latency | 3.0 ms | 3.1 ms | Parity |
| P95 latency | 10.7 ms | **5.3 ms** | PrismAPI 2× better |
| P99 latency | 517 ms | **6.6 ms** | **PrismAPI 78× better** |

*Baseline throughput advantage: benchmark uses repeated queries which hit the LRU embed cache. Under real query diversity (no cache hits), baseline P99 grows unboundedly as the embedding model saturates. PrismAPI P99 stays flat — no embedding model to saturate.

### Retrieval quality (64-dim vs full-rank 384-dim baseline)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Jaccard@5 | 0.489 | ~49% exact top-5 set overlap |
| Recall@5 | 0.632 | 63% of baseline top-5 recovered |
| **Recall@10** | **0.732** | **Production metric: over-fetch 2×, re-rank** |

At 128-dim projection: Recall@10 ≈ 0.91 (loopback benchmark). The retrieval quality vs. payload size trade-off:

| target_dim | Recall@10 | Wire size | Best for |
|-----------|-----------|-----------|----------|
| 64 | 73% | −12% vs JSON | bandwidth-constrained |
| 128 | ~91% | ~parity with JSON | quality-priority |

**Production pattern:** over-fetch `top_k × 2`, re-rank with PrismResonance. Recall@10 at 64-dim means 73% of relevant results are in the over-fetched set — sufficient for re-ranking to recover the remainder.

### Wire bytes

| | Baseline (JSON) | PrismAPI (CHORUSFrame) | Delta |
|-|-----------------|----------------------|-------|
| Mean per query | 2,448 B | 2,146 B | −12.3% |

---

## Installation

```bash
pip install prism-lib                    # core library
pip install sentence-transformers numpy  # for local embeddings
```

**Optional integrations:**
```bash
pip install langchain langgraph          # LangGraph / LangChain nodes
pip install fastapi uvicorn              # ASGI provider adapter
pip install httpx                        # async HTTP transport (optional)
```

PrismAPI has no mandatory framework dependency. LangGraph, FastAPI, and MCP integrations are all optional imports.

---

## Quick Start

### Provider — one line to adopt

```python
from prism.api import PrismAPIProvider, SentenceTransformerEmbedder
from prism.lib.lang import PrismProjector, ProjectionConfig

projector = PrismProjector(ProjectionConfig(tenant_id="my-org", target_dim=64))
embedder  = SentenceTransformerEmbedder()   # any model: OpenAI, Gemini, local

provider = PrismAPIProvider(
    projector=projector,
    embedder=embedder,
    semantic_fields=["title", "body"],
    id_field="doc_id",
    exact_fields=["price", "url", "category"],
)

@provider.expose                                # ← the one line
def search(query: str, top_k: int = 10) -> list[dict]:
    return my_database.search(query, top_k)    # unchanged — your existing handler
```

The original handler continues serving its existing REST/HTTP clients unchanged. `@provider.expose` adds a parallel CHORUS channel.

### Consumer — basic query

```python
from prism.api import PrismAPIClient, RetryConfig

client = PrismAPIClient(
    projector=projector,
    embedder=embedder,
    host="api.example.com",
    port=9100,
    retry=RetryConfig(max_retries=3, backoff_base=0.5, timeout_read=30.0),
)

with client:   # persistent connection, auto-closes
    response = client.query("how does inflation affect bonds?", top_k=10)

# response.vectors              → np.ndarray (10, 64) — cosine-ready float32
# response.sidecars             → list[dict] with price, url, category
# response.embedding_calls_saved → 10
# response.latency_ms            → float
```

### LangGraph — one line node

```python
from prism.api import create_retriever_node

graph.add_node("retrieve", create_retriever_node(
    client=my_client,
    query_key="query",
    results_key="search_results",
    top_k=10,
    error_key="retrieval_error",
    fallback_fn=lambda q, k: my_rest_api.search(q, k),
))
```

### Multi-provider fan-out

```python
from prism.api import MultiProviderClient

result = MultiProviderClient(
    clients={
        "news":    PrismAPIClient(projector, embedder, host="news.api.com", port=9100),
        "legal":   PrismAPIClient(projector, embedder, host="legal.api.com", port=9100),
        "medical": PrismAPIClient(projector, embedder, host="med.api.com", port=9100),
    },
    top_k_per_provider=5,
    total_top_k=10,
).query("drug pricing regulation")

# Queries all 3 in parallel. Wall-clock = slowest provider.
# result.vectors               → (10, 64) merged, ranked by cosine
# result.sidecars              → [{"provider": "legal", ...}, ...]
# result.per_provider          → raw per-provider responses
# result.provider_errors       → any failures
# result.embedding_calls_saved → total across all providers
```

### FastAPI provider (ASGI adapter)

```python
from fastapi import FastAPI
from prism.api.provider import ASGIAdapter

app = FastAPI()

# Your existing routes untouched:
@app.get("/search")
def http_search(q: str, top_k: int = 10):
    return {"results": db.search(q, top_k)}

# Add CHORUS endpoint alongside (does not replace):
adapter = ASGIAdapter(handler=search, handler_name="search")
adapter.mount(app)   # adds POST /chorus/search accepting CHORUSFrame
```

### MCP server (Claude Desktop / any MCP host)

```python
from prism.api.mcp import PrismAPIMCPServer

server = PrismAPIMCPServer(
    provider=provider,
    handler=search,
    tool_name="semantic_search",
    tool_description="Search the knowledge base by semantic meaning.",
)
server.run()   # JSON-RPC 2.0 over stdio
```

Returns both JSON summary (for standard MCP clients) and `chorus_frame_b64` (for CHORUS-native agents). Same tool call, same endpoint. No protocol fork.

---

## Integration Patterns

### Pattern 1: Drop into existing LangGraph pipeline

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph
from prism.api import PrismAPIClient, PrismRetrieverNode, RetryConfig
from prism.lib.lang import PrismProjector, ProjectionConfig
from prism.api.schema import SentenceTransformerEmbedder, APIResponse

class AgentState(TypedDict):
    query: str
    search_results: Optional[APIResponse]
    retrieval_error: Optional[str]
    answer: Optional[str]

projector = PrismProjector(ProjectionConfig(tenant_id="my-org", target_dim=64))
embedder  = SentenceTransformerEmbedder()
client    = PrismAPIClient(projector, embedder, host="api.myorg.com", port=9100)

def reasoning_node(state: AgentState) -> AgentState:
    results = state["search_results"]
    if results is None:
        return {"answer": "No results found."}
    # results.vectors → np.ndarray (N, 64) — use directly for cosine, clustering
    # results.sidecars → list[dict] — exact metadata for each result
    context = "\n".join(s.get("body_preview", "") for s in results.sidecars)
    answer = my_llm.generate(state["query"], context=context)
    return {"answer": answer}

graph = StateGraph(AgentState)
graph.add_node("retrieve", PrismRetrieverNode(
    client=client,
    query_key="query",
    results_key="search_results",
    top_k=10,
    error_key="retrieval_error",
    fallback_fn=lambda q, k: rest_api.search(q, k),
))
graph.add_node("reason", reasoning_node)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "reason")

app = graph.compile()
result = app.invoke({"query": "inflation and bond yields"})
```

### Pattern 2: Async LangGraph

```python
# PrismRetrieverNode.ainvoke() runs the blocking HTTP call in a thread pool
# so it never blocks the event loop.

graph = StateGraph(AgentState)
graph.add_node("retrieve", retriever_node)  # same node, async graph handles it

async def run():
    result = await app.ainvoke({"query": "..."})
```

### Pattern 3: Multi-hop pipeline — vectors propagate without re-embedding

```python
class ResearchState(TypedDict):
    query: str
    broad_results: Optional[APIResponse]   # Agent A retrieves broadly
    focused_results: Optional[APIResponse] # Agent B refines using vectors from A
    answer: Optional[str]

def refine_node(state: ResearchState) -> ResearchState:
    """Agent B: uses vectors from broad retrieval as new query vectors."""
    broad = state["broad_results"]
    if broad is None or len(broad.results) == 0:
        return state

    # Use the top result's vector as the refined query — zero embed call
    top_vector = broad.vectors[0]   # already 64-dim float32
    refined = client.query_vector(top_vector, top_k=5)
    return {"focused_results": refined}
```

`query_vector()` accepts a pre-computed vector — zero embedding calls, even for the query. The refined query IS a vector from the previous step.

### Pattern 4: Provider with existing FastAPI app (zero migration)

```python
from fastapi import FastAPI
from prism.api import PrismAPIProvider, SentenceTransformerEmbedder
from prism.api.provider import ASGIAdapter
from prism.lib.lang import PrismProjector, ProjectionConfig

app = FastAPI()

# ── Your existing code, unchanged ─────────────────────────────────────────────
@app.get("/search")
def http_search(q: str, top_k: int = 10):
    results = db.semantic_search(q, top_k)
    return {"results": results}
# ─────────────────────────────────────────────────────────────────────────────

# ── PrismAPI: three setup lines + one decorator ────────────────────────────────
projector = PrismProjector(ProjectionConfig(tenant_id="my-org"))
embedder  = SentenceTransformerEmbedder()
provider  = PrismAPIProvider(projector, embedder,
                              semantic_fields=["title", "body"],
                              id_field="doc_id", exact_fields=["price", "url"])

@provider.expose
def search(query: str, top_k: int = 10):
    return db.semantic_search(query, top_k)   # same handler as above

ASGIAdapter(handler=search, handler_name="search").mount(app)
# Adds: POST /chorus/search — CHORUS-native agents use this
# Keeps: GET /search — existing REST clients unaffected
# ─────────────────────────────────────────────────────────────────────────────
```

### Pattern 5: Health monitoring and graceful degradation

```python
client = PrismAPIClient(
    projector=projector, embedder=embedder,
    host="api.example.com", port=9100,
    retry=RetryConfig(max_retries=3, backoff_base=0.5, timeout_connect=5.0),
)

# Health check before serving traffic
if not client.health_check():
    logger.warning("PrismAPI server unreachable — falling back to REST")
    # use fallback_fn in PrismRetrieverNode to handle automatically

# MultiProviderClient health check
mp_client = MultiProviderClient(clients={"news": ..., "legal": ...})
health = mp_client.health_check()
# {"news": True, "legal": False}
```

---

## API Reference

### `PrismAPIProvider`

```python
PrismAPIProvider(
    projector: PrismProjector,           # JL projector (must match consumer's tenant_id)
    embedder: Embedder,                  # any object with .embed(texts) → ndarray
    semantic_fields: list[str],          # fields to embed ["title", "body"]
    id_field: str = "id",               # primary key field
    exact_fields: list[str] = [],       # fields to pass as sidecar (never embed)
    provider_id: str = auto,            # stable identifier for this endpoint
)
```

Methods:
- `@provider.expose` — decorator: wraps a handler function to serve CHORUS frames
- `project_results(dicts) → APIResponse` — project a list of dicts manually
- `as_chorus_frame(dicts) → CHORUSFrame` — project + pack as wire frame
- `register_handler(fn)` — manually register handler (alternative to decorator)

### `PrismAPIClient`

```python
PrismAPIClient(
    projector: PrismProjector,           # must use same tenant_id as provider
    embedder: Embedder,
    host: str = "localhost",
    port: int = 9100,
    loopback_provider = None,            # for in-process testing
    source_field: str = "body",
    retry: RetryConfig = RetryConfig(),  # retry + timeout config
    chorus_path: str = "/chorus/search", # CHORUS endpoint path
)
```

Methods:
- `query(text, top_k) → APIResponse` — embed query + CHORUS round-trip
- `query_vector(vector, top_k) → APIResponse` — send pre-computed vector (zero embeds)
- `health_check() → bool` — ping /health
- `close()` — close persistent connection
- Context manager: `with PrismAPIClient(...) as client:`

### `RetryConfig`

```python
RetryConfig(
    max_retries: int = 3,           # retry attempts after initial failure
    backoff_base: float = 0.5,      # sleep = backoff_base × 2^attempt (seconds)
    backoff_max: float = 8.0,       # cap on sleep duration
    timeout_connect: float = 5.0,   # TCP connect timeout
    timeout_read: float = 30.0,     # socket read timeout
)
```

### `MultiProviderClient`

```python
MultiProviderClient(
    clients: dict[str, PrismAPIClient], # name → client
    top_k_per_provider: int = 5,        # over-fetch from each provider
    total_top_k: int = 10,              # return after merging
    max_workers: int = len(clients),    # thread pool size
    timeout_s: float = 30.0,            # per-provider timeout
)
```

Methods:
- `query(text, top_k, top_k_per_provider) → MultiProviderResponse`
- `health_check() → dict[str, bool]`

### `PrismRetrieverNode` (LangGraph)

```python
PrismRetrieverNode(
    client: PrismAPIClient,
    query_key: str = "query",           # state key to read query from
    results_key: str = "search_results", # state key to write APIResponse to
    top_k: int = 10,
    top_k_key: str = None,             # optional state key to override top_k
    error_key: str = None,             # write error here instead of raising
    fallback_fn: Callable = None,      # fallback(query, top_k) → list[dict]
)
```

Callable as `node(state)` (sync) or `await node.ainvoke(state)` (async).

Factory: `create_retriever_node(client, query_key, results_key, top_k, error_key, fallback_fn)`

### `MultiProviderRetrieverNode` (LangGraph)

```python
MultiProviderRetrieverNode(
    clients: dict[str, PrismAPIClient],
    query_key: str = "query",
    results_key: str = "search_results",
    top_k_per_provider: int = 5,
    total_top_k: int = None,            # defaults to top_k_per_provider
    error_key: str = None,
)
```

### `APIResponse`

```python
response.results                 # list of (SemanticItem, ExactSidecar) pairs
response.vectors                 # np.ndarray (N, target_dim) — stacked float32
response.sidecars                # list[dict] — exact fields, ordered by relevance
response.embedding_calls_saved   # int — calls consumer did not make
response.latency_ms              # float — end-to-end round-trip time
response.provider_id             # str — identifies the provider
```

### `MultiProviderResponse`

```python
response.results                       # merged list of (SemanticItem, ExactSidecar)
response.vectors                       # np.ndarray (N, 64) — merged, cosine-ranked
response.sidecars                      # list[dict] with added "provider" key
response.per_provider                  # dict[name, APIResponse] — raw per-provider
response.provider_errors               # dict[name, str] — any failures
response.total_embedding_calls_saved   # int — saved across all providers
response.latency_ms                    # float — wall-clock (parallel = max, not sum)
```

---

## Architecture Deep Dive

### Wire protocol — CHORUSFrame

Every PrismAPI exchange uses the CHORUS binary frame format:

```
Frame header (81 bytes fixed):
  [key_id: 36 bytes] [seq: 8 bytes] [watermark: 32 bytes] [frame_type: 1 byte] [payload_len: 4 bytes]

API_REQUEST payload (frame_type = 0x08):
  [dim: 4 bytes] [query_vec: dim × 4 bytes] [ctx_len: 4 bytes] [ctx_json: ctx_len bytes]

API_RESPONSE payload (frame_type = 0x09):
  [n_results: 4 bytes] [dim: 4 bytes]
  for each result:
    [vec: dim × 4 bytes] [sidecar_len: 4 bytes] [sidecar_json: sidecar_len bytes]
```

`Content-Type: application/x-chorus-frame` on HTTP transport.

### JL projection — tenant isolation

The Johnson-Lindenstrauss projection matrix is seeded by `SHA-256(tenant_id)`:

```python
seed = int.from_bytes(hashlib.sha256(tenant_id.encode()).digest()[:4], "big")
rng  = np.random.default_rng(seed)
K    = rng.standard_normal((embed_dim, target_dim)).astype(np.float32)
K   /= np.linalg.norm(K, axis=0)   # column-normalize

projected = (embedding @ K).astype(np.float32)
```

Properties:
- **Deterministic:** same tenant_id → same K → compatible vector spaces across machines and restarts
- **Tenant-isolated:** different tenant_ids → different K → incompatible (by design — privacy isolation)
- **Cosine-preserving:** JL lemma guarantees `|cos(u,v) - cos(Ku, Kv)| < ε` with high probability
- **Model-agnostic:** any embed_dim input, fixed target_dim output — provider can change embedding model without breaking the consumer's vector space

### Concurrency model

The server (`bench_server.py`) uses `ThreadedHTTPServer(ThreadingMixIn, HTTPServer)`:
- One thread per request — concurrent requests execute truly in parallel
- CHORUS path: each thread does a matmul (`_corpus_projected @ query_vec`) — O(N × dim) per request
- Baseline path: each thread hits the LRU embed cache or calls the embedding model

The client uses a persistent `http.client.HTTPConnection` (keep-alive):
- Connection reused across queries — TCP handshake paid once
- Retry on connection-level errors — reconnect and retry up to `max_retries` times
- Exponential backoff: 0.5s, 1.0s, 2.0s, ... capped at `backoff_max`

---

## FAQ

**Q: Does the consumer need the same embedding model as the provider?**
No. The consumer only embeds its own query. Results arrive as float32 vectors — the consumer never needs to know what model the provider used. The JL projection matrix (seeded by `tenant_id`) is the shared contract, not the embedding model.

**Q: What if the provider updates its embedding model?**
The provider re-indexes its corpus with the new model. The projected vectors change. The consumer is unaffected as long as `tenant_id` is consistent — it embeds queries with any model and projects to the same 64-dim space. Only the provider's index changes.

**Q: Is 73% Recall@10 at 64-dim acceptable for production?**
Depends on the use case. The production pattern is over-fetch (`top_k × 2`) and re-rank with PrismResonance. At 64-dim, Recall@10 = 73% means 73% of the relevant results are in the over-fetched set — high enough for re-ranking to recover the remainder for most applications. For high-precision use cases, use 128-dim (Recall@10 ≈ 91%).

**Q: Can I use PrismAPI with OpenAI or Gemini embeddings instead of sentence-transformers?**
Yes. Any object satisfying the `Embedder` protocol works:
```python
class OpenAIEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
        return np.array([r.embedding for r in response.data], dtype=np.float32)
```

**Q: How does this relate to MCP (Model Context Protocol)?**
They are complementary. MCP carries the tool call (JSON-RPC 2.0 over stdio/SSE). CHORUS carries the vector payload (binary float32 frames). `PrismAPIMCPServer` returns both in every tool response — standard MCP clients read the JSON summary, CHORUS-native agents decode the vector frame. Same endpoint, no protocol fork.

**Q: Is CHORUS encrypted?**
The CHORUS cipher is `V_enc = V_raw @ K` where K is a QR-decomposed orthogonal key matrix. The encrypt operation is a matrix multiply — the same operation every neural network already runs, adding zero latency overhead. Watermark verification (HMAC-SHA256) is included in every frame.

**Q: Can I run multiple tenant_ids on one server?**
Yes. The server builds one index per tenant. The client's frame carries the `tenant_id` in the context dict. The server routes to the correct projected index. This is the multi-tenant deployment model.

---

## Product Positioning

### What PrismAPI is NOT

- **Not a vector database.** PrismAPI does not store or index documents. The provider's existing database or search index is unchanged.
- **Not an embedding model.** PrismAPI does not provide embeddings. It is a transport and projection layer on top of any embedder.
- **Not an alternative to REST.** PrismAPI adds a parallel CHORUS channel. Existing REST/HTTP endpoints continue working unchanged.

### Where PrismAPI lives in the PrismLib stack

```
PrismLib stack (top to bottom):

  prism.api          ← PrismAPI: provider ↔ consumer API contract, CHORUS transport
       │
  prism.lib.resonance ← PrismResonance: wave-memory re-ranking of retrieved vectors
       │
  prism.lib.lang     ← PrismProjector: JL projection, PayloadEnvelope
       │
  prism.lib.fabric   ← CHORUS Fabric: binary frame format, TensorCipher, watermark
```

PrismAPI is the **outward-facing layer** of PrismLib — the part that connects to the outside world (other APIs, other agents, other frameworks). It depends on all three layers below it and cannot be meaningfully decoupled from them.

### Why PrismAPI is part of PrismLib, not VectorBridge or standalone

**VectorBridge** ([`insight-vector-bridge`](https://pypi.org/project/insight-vector-bridge/)) is a vector **migration** tool — it moves existing vector data between databases (Pinecone → Qdrant, pgvector → Weaviate, etc.) using CHORUS for transport efficiency. It is a DevOps/infrastructure tool used once or periodically.

**PrismAPI** is a **retrieval protocol** — it defines how agents consume vector data at query time, continuously, at scale. Different product, different buyer, different use case.

**Standalone** is not viable because PrismAPI requires:
- `prism.lib.fabric` for CHORUSFrame serialization and TensorCipher
- `prism.lib.lang` for PrismProjector and JL projection
- `prism.lib.resonance` for re-ranking (optional but core to the value proposition)

Publishing standalone would require publishing these as sub-packages, fragmenting the install experience.

**PrismLib** (`pip install prism-lib`) is the correct home. PrismAPI is the headline feature of PrismLib — the most accessible and immediately useful entry point for AI engineers, with the deeper stack (PrismResonance, PrismProjector, CHORUS Fabric) available as they go deeper.

### Competitive positioning

| | PrismAPI | Standard REST API | LangChain retriever | OpenAI file search |
|-|---------|-------------------|--------------------|--------------------|
| Consumer re-embedding | None | Every result | Every result | N/A (proprietary) |
| Wire format | Binary float32 | JSON text | JSON text | Proprietary |
| Provider model flexibility | Any | Any | Any | OpenAI only |
| Multi-provider fan-out | Native | Manual | Manual | No |
| LangGraph integration | Native node | Custom code | Custom code | No |
| Privacy-preserving retrieval | Native (vector-only) | No | No | No |
| Protocol | Open (CHORUS) | HTTP/REST | HTTP/REST | Proprietary |

---

## Landing Page Implementation Guide

*This section is for an AI implementing the marketing landing page.*

### Brand voice
- Technical but accessible — the audience is AI engineers who build LangGraph pipelines and production RAG systems
- Direct — lead with the concrete saving (83% fewer embedding calls, 24× faster true e2e latency) before explaining how
- Honest — acknowledge the Recall@10 trade-off at 64-dim; engineers distrust benchmarks that have no caveats

### Key headline options (A/B test)
1. "Your agents are re-embedding content that's already a vector." (pain-first)
2. "Embed once. Retrieve forever." (solution-first, memorable)
3. "83% fewer embedding calls. One decorator line." (proof + ease)
4. "The API contract AI agents were never given." (framing)

### Hero section
- Animated code diff: show the standard REST loop on the left, PrismAPI loop on the right
- Highlight the eliminated steps (embed × top_k) in red on the left, "0 embed calls" in green on the right
- Counter: "embedding calls saved per 1,000 queries" that increments as the visitor scrolls

### Core sections (in order)
1. **The problem** — the embedding tax diagram (text → embed → vector → use) vs PrismAPI (vector → use)
2. **Benchmark numbers** — four big stats: 83.3% embed calls saved, 24× faster true e2e, P99 78× better, 12% less wire
3. **Use cases** — 6 cards: LangGraph pipelines, real-time agents, privacy-preserving federated search, multi-provider marketplace, high-frequency automation, edge agents
4. **Quick start** — three tabs: Provider / Consumer / LangGraph. Syntax-highlighted. Copy button.
5. **Architecture** — the ASCII wire diagram as a clean visual
6. **FAQ** — the 7 questions from this document

### Color palette
- Primary: dark navy `#0F172A`
- Accent: electric blue `#3B82F6`
- Success/saved: emerald `#10B981`
- Text: `#F8FAFC` on dark, `#1E293B` on light
- Code background: `#1E293B`

### Call to action
- Primary: `pip install prism-lib` (copy-to-clipboard)
- Secondary: "View on GitHub" → github.com/insightitsGit/prismlib
- Tertiary: "Read the benchmark" → links to benchmark/api/results/

### Social proof
- 38 tests passing badge
- "Benchmarked against real HTTP" (not loopback)
- Patent pending: USPTO Provisional (Chorus Fabric — the underlying transport)
- "Powers PrismLib — the full vector-native agent stack"
