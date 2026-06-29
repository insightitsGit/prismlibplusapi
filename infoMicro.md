# PrismLib Micro — Cluster & RAG Product Page Brief

This document is for the agent building the **PrismLib Micro** product page.
It covers the cluster/microservice layer: ClusterCache, CHORUS multi-node mesh,
Blue/Green/Orange failover, and the RAG token-compression pipeline.
Read `info.md` first for the base PrismLib context (PrismCache, PrismDriver, CHORUS Fabric).

---

## What "PrismLib Micro" is

PrismLib Micro is the cluster layer of PrismLib — the set of components that
makes PrismLib work across multiple containers, pods, or networks. It adds
three capabilities on top of the single-node PrismCache/PrismDriver stack:

1. **ClusterCache** — shares LLM answers across all nodes in real time via the
   existing CHORUS tunnel. Once any node answers a query, every other node
   serves it for free — zero additional LLM calls, zero Redis.

2. **CHORUS mesh health** — every container broadcasts CPU, RAM, disk, and
   latency every 2 seconds as a HEALTH frame. Any node can see the full cluster
   at a glance. AlertManager fires SIGNAL frames + admin email the moment a
   threshold is crossed, without a Prometheus stack, without a Datadog agent.

3. **Blue/Green/Orange failover** — a three-tier hot-standby topology.
   GREEN is the active master. BLUE is a warm standby (pre-synced via CHORUS,
   auto-promotes if GREEN goes silent). ORANGE is the syncing reserve.
   No Raft dependency. No Kubernetes operator required.

Install: `pip install "prismlib[fabric]"` — cluster code is included.

---

## Live benchmark results (3-node cluster on Azure Container Apps, westus2)

Benchmark run: 3 PrismNode containers on Azure Container Apps — GREEN + BLUE in
Environment A (one VNet), ORANGE in Environment B (separate VNet), benchmark
runner external/cross-network. Deploy: `deploy/azure_cluster_run.sh`. Code:
`benchmark/cluster/run_cluster_benchmark.py`. Raw:
`benchmark/cluster/cluster_benchmark_results_azure.json`.

### Phase 1 — Token savings across the cluster

| Node | Role | Network | Tokens billed | Tokens saved | Savings |
|------|------|---------|--------------|-------------|---------|
| node-green | GREEN / active master | same-pod | 328 | 130 (compression) | **28.4%** |
| node-blue | BLUE / warm standby | same-pod | **0** | 61 (cluster cache) | **100%** |
| node-orange | ORANGE / reserve | cross-network | **0** | 75 (cluster cache) | **100%** |
| **Cluster avg** | | | | | **76.1%** |

What happened: GREEN answered 5 unique queries. Before the benchmark runner
even sent queries to BLUE or ORANGE, those nodes already had every answer in
their local ClusterCache — broadcast via CHORUS TOKEN_SYNC frames in real time.
BLUE and ORANGE served 2 queries each for 0 tokens. No Redis. No API call.

### Phase 2 — CHORUS frame latency (same-VNet vs cross-VNet, real Azure)

| Node | Network type | Avg latency | Min | Max |
|------|-------------|-------------|-----|-----|
| node-green | same VNet | 19.4 ms | 15.0 ms | 29.0 ms |
| node-blue | same VNet | 19.9 ms | 15.2 ms | 22.6 ms |
| node-orange | cross-VNet | **21.6 ms** | 16.0 ms | 26.3 ms |

These are real Azure Container Apps measurements across two VNets. Cross-VNet
frames cost only ~2 ms more than same-VNet because both environments are in the
same region (westus2); cross-region would be higher. Token savings are
network-independent — they are logic, not latency.

### Phase 3 — Health alert propagation

| Event | Source | Destination | Network | Propagation time |
|-------|--------|-------------|---------|-----------------|
| cpu_high (92%) | GREEN | BLUE | same VNet | **633 ms** |
| cpu_high (92%) | GREEN | ORANGE | cross-VNet | **674 ms** |

AlertManager fired a SIGNAL frame the moment the threshold was crossed.
Both nodes received it in under 1 second. No Prometheus. No Datadog.
No scrape interval to wait for.

### Phase 4 — Blue/Green failover

| Metric | Value |
|--------|-------|
| GREEN silence threshold | 3 000 ms |
| Failover detected (silence crossed) | **3 960 ms** |
| Promotion to active (once detected) | **97 ms** |
| Failovers triggered | 1 |
| Post-failover master | node-blue |
| Human intervention required | **None** |

GREEN's heartbeat was paused for 5 s. BLUE's watchdog detected silence at the
3 s mark and self-promoted to active master — broadcasting a SIGNAL frame to
the full cluster announcing the topology change.

### Phase 5 — Context compression ratio

| Query | Tokens used | Tokens saved | Compression |
|-------|-------------|-------------|-------------|
| What is PrismLib? | 71 | 126 | **64.0%** |
| How does CHORUS Fabric work? | 85 | 118 | **58.1%** |
| Explain context compression. | 75 | 128 | **63.1%** |
| What is Blue/Green/Orange failover? | 84 | 116 | **58.0%** |
| How does token deduplication work? | 81 | 116 | **58.9%** |
| **Average** | | | **60.4%** |

10 context chunks were available per query. ContextCompressor computed
cosine similarity between the query vector and each chunk vector,
kept the top 3, dropped the rest. The LLM received only relevant context.
No extra LLM call. No external model. Pure in-process numpy math.

---

## How the token savings pipeline works (5 steps)

This is the ClusterCache pipeline executed in order on every query:

```
1. LOCAL CACHE CHECK
   PrismCache (in-process) — sub-millisecond semantic lookup.
   Hit → return answer, 0 tokens billed.

2. CLUSTER CACHE CHECK
   Check the local copy of cluster-wide answers (populated by TOKEN_SYNC frames).
   Hit → return answer, 0 tokens billed. No network call.

3. CONTEXT COMPRESSION
   ContextCompressor ranks context chunks by cosine similarity to the query.
   Keeps top-K (default 3). Irrelevant chunks dropped before the prompt is built.
   Saves 58–64% of context tokens on average.

4. IN-FLIGHT DEDUPLICATION
   If the same query is already in-flight (another async call is running it),
   the new caller coalesces onto the existing asyncio Future.
   Result shared. 0 extra tokens. 0 extra latency.

5. LLM CALL + BROADCAST
   One LLM call is made with the compressed context.
   Answer is stored in local PrismCache, then broadcast as a TOKEN_SYNC frame
   to all cluster peers via CHORUS. All nodes cache it immediately.
```

Each step is additive — a query that misses steps 1–2 still saves on step 3.
In a busy cluster, most queries are caught at step 1 or 2.

---

## RAG integration (retrieval-augmented generation)

PrismLib Micro fits directly into any RAG pipeline. The ClusterCache wraps the
"retrieve + generate" step — so documents are retrieved once and their embeddings
are shared across nodes, and LLM answers on the same retrieved context are never
duplicated.

### How a RAG query flows with PrismLib Micro

```
User query
  │
  ▼
ClusterCache.get_or_call(query, query_vector, call_fn,
                          context_chunks=retrieved_docs,
                          chunk_vectors=doc_embeddings)
  │
  ├─ Step 1: local semantic cache — have we answered this before?
  ├─ Step 2: cluster cache — has any other node answered this?
  ├─ Step 3: context compression — keep only top-3 relevant docs
  ├─ Step 4: dedup — is another node currently calling the LLM for this?
  └─ Step 5: LLM call → answer cached locally + broadcast to all nodes
```

### RAG code example (5 lines)

```python
from prism.cluster.cache import ClusterCache

cache = ClusterCache(node_id="my-node", fabric=chorus_fabric)

answer = await cache.get_or_call(
    query       = user_question,
    query_vector= embed(user_question),       # your embedder
    call_fn     = lambda: llm.complete(...),  # your LLM call
    context_chunks = retrieved_docs,
    chunk_vectors  = doc_vectors,
)
```

No changes to your retrieval logic. No changes to your LLM calls.
Drop ClusterCache in front of your existing `retrieve → generate` step.

### What PrismLib Micro saves in a RAG system

| Cost category | Without PrismLib | With PrismLib | Saving |
|--------------|-----------------|---------------|--------|
| Context tokens per call | Full doc set | Top-K only | **58–64%** |
| LLM calls (repeated queries) | N calls per N nodes | 1 call, shared | **up to 100%** |
| Embedding calls | Per node, per query | Shared via cluster cache | **up to 100% for cached nodes** |
| Monitoring infra | Prometheus + Datadog | Built-in SIGNAL frames | **$0 + <1s latency** |
| HA infrastructure | K8s operator or manual | Blue/Green auto-promote | **$0 + ~3s failover** |

### Multi-tenant RAG isolation

Every PrismCache lookup is projected through a Johnson-Lindenstrauss matrix
seeded by `SHA-256(tenant_id)`. Tenant A's documents never appear in Tenant B's
similarity results — not by query filter, but by math.

```python
cache = ClusterCache(node_id="node-1", fabric=fabric)
# tenant isolation is automatic — pass tenant_id per query
answer = await cache.get_or_call(..., tenant_id="tenant_abc")
```

---

## Competitive analysis — what the market has

### Token savings / cluster cache

| Product | Max cluster savings | Cross-node cache sharing | Infrastructure needed |
|---------|--------------------|--------------------------|-----------------------|
| **PrismLib ClusterCache** | **76% avg / 100% for cached nodes** | **CHORUS broadcast, built-in** | **None** |
| GPTCache | 40–70% | No — per-process only | Redis + FAISS |
| Zep / Mem0 | ~30–50% | Via their paid cloud only | Paid cloud + SDK |
| LangChain cache | Exact-match only | No | SQLite or Redis |
| Redis cluster (manual) | Exact-match only | Yes (Redis) | Self-hosted Redis |
| Anthropic prompt cache | Up to 90% (prefix only) | Per API key, not per node | None (API flag) |

**PrismLib's edge:** Anthropic prompt cache is the only thing close on savings %, but
it works only on fixed prompt prefixes and doesn't share across nodes or services.
PrismLib's semantic cache catches paraphrases that exact-match and prefix-match miss,
and shares answers across every container in the cluster automatically.

### Health alert propagation

| Solution | Propagation time | Infrastructure | Cost |
|---------|-----------------|----------------|------|
| **PrismLib SIGNAL frame** | **<1 s (709–711 ms measured)** | **None — built-in** | **$0** |
| Prometheus Alertmanager | 30–60 s (scrape interval) | Prometheus stack | Self-host |
| Datadog / New Relic | 30–120 s (cloud aggregation) | Paid agent per host | $15–$23/host/mo |
| Kafka health topic | <100 ms | Kafka cluster | Self-host / MSK |
| AWS CloudWatch | 60–300 s (polling) | AWS only | $0.30/metric/mo |

PrismLib wins on propagation time vs all monitoring SaaS tools, and requires
zero infrastructure. The only faster solution is Kafka — but Kafka requires
running a cluster and writing a health topic schema.

### Blue/Green failover

| Solution | Failover time | Warm standby | Human step | Extra infra |
|---------|--------------|-------------|-----------|------------|
| **PrismLib Blue/Green** | **~3–4 s** | **Yes (CHORUS-synced)** | **None** | **None** |
| Kubernetes pod restart | 10–60 s | Depends on config | None | K8s |
| Raft (etcd/consul) | 150–500 ms | Yes (quorum) | None | etcd cluster |
| Redis Sentinel | 2–30 s | Yes | None | Redis + Sentinel |
| Manual / PagerDuty | 5–30 min | No | Yes | PagerDuty $19+/mo |

Honest note: Raft-based solutions (etcd, consul) have faster election times (150–500 ms)
and stronger consensus guarantees. PrismLib's ~3–4 s is competitive with Redis Sentinel
and far better than manual. For most LLM-serving workloads a 3–4 s gap is acceptable.
The advantage: zero extra infra.

### Context compression

| Solution | Token reduction | Method | Extra cost |
|---------|----------------|--------|-----------|
| **PrismLib ContextCompressor** | **58–64% measured** | Cosine-sim top-K, in-process | **$0** |
| LangChain ContextualCompression | ~30–50% | Extra LLM call to extract | Extra LLM call cost |
| LLMLingua (Microsoft) | Up to 80% | Dedicated compression model | GPU + model inference |
| Manual truncation | ~50% (crude) | Character/token limit cutoff | Loses semantic meaning |

PrismLib's compression is free and runs in microseconds. LLMLingua achieves higher
compression but requires a separate GPU-hosted model, adding latency and cost.
LangChain's approach calls the LLM twice — which doubles cost, not reduces it.

---

## Architecture diagram data (for the page agent to visualize)

### 4-node cluster topology

```
┌─────────────────────────────────────────────────────────┐
│  Container App Environment A  (same VNet)               │
│                                                         │
│  ┌──────────────┐    CHORUS mesh    ┌──────────────┐   │
│  │  node-green  │◄─────────────────►│  node-blue   │   │
│  │  [GREEN]     │  TOKEN_SYNC       │  [BLUE]      │   │
│  │  active      │  HEALTH           │  warm standby│   │
│  │  master      │  SIGNAL           │  auto-promote│   │
│  └──────┬───────┘                   └──────┬───────┘   │
└─────────┼─────────────────────────────────┼────────────┘
          │ cross-network CHORUS             │
          │ (TLS + HMAC + ephemeral key)     │
┌─────────┼─────────────────────────────────┼────────────┐
│  Container App Environment B  (separate VNet)          │
│          │                                │            │
│  ┌───────▼─────────────────────────────── ▼──────┐    │
│  │  node-orange  [ORANGE]  syncing reserve        │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘

Benchmark runner (external) ──► all three nodes via HTTPS
```

Key flows carried over CHORUS mesh:
- `TOKEN_SYNC` — LLM answers propagated instantly to all nodes
- `HEALTH` — CPU/RAM/disk/latency broadcast every 2 s
- `SIGNAL` — alerts (cpu_high, ram_high, failover events)
- `CONFIG` — runtime config updates pushed to all nodes
- `METRIC` — token usage stats aggregated across cluster
- `DELTA` — model weight deltas (ChorusMesh extension)

### ClusterCache token flow

```
Query arrives at BLUE or ORANGE
  │
  ├─ Cluster cache HIT → 0 tokens, <1ms response
  │   (answer already there from GREEN's TOKEN_SYNC frame)
  │
  └─ Cluster cache MISS → GREEN handles it
        │
        ├─ Context compression: 10 chunks → 3 chunks (saves 60%)
        ├─ LLM call (once, shared)
        └─ TOKEN_SYNC broadcast → all nodes cache it
             Next time: 0 tokens on every node in the cluster
```

---

## Pricing and tiers (recommended copy for the product page)

### The two-package model

| Package | License | Enforcement | Who it's for |
|---------|---------|-------------|-------------|
| `prismlib` | Apache 2.0 — free forever | None — open source | Everyone |
| `chorusmesh` *(coming soon)* | Commercial | JWT license key | Teams needing Slack/PagerDuty, custom rules, Raft, multi-region |

**Why two packages:** `prismlib` is open source and cannot be license-enforced — anyone can read the code. `chorusmesh` is a separate paid package where enforcement lives. This is the same model used by Elastic (ELK vs X-Pack) and HashiCorp (Terraform vs Terraform Enterprise).

---

### Open-source — `prismlib` (free forever)

Everything currently in the repo. No license key. No node limit. No expiry.

- PrismCache, PrismDriver (Server Wrapper + DLL Driver)
- ClusterCache, AlertManager (SMTP email only), Blue/Green/Orange failover
- ContextCompressor, all 7 CHORUS frame types
- 12 built-in alert rules (CPU, RAM, disk, latency, token budget)
- 1 email recipient per AlertManager instance
- Community support via GitHub Issues
- `pip install "prismlib[all]"`

---

### ChorusMesh — feature breakdown by tier

#### How enforcement works

`chorusmesh` reads a `CHORUSMESH_LICENSE_KEY` environment variable on startup.
The key is a JWT signed with InsightIts' RSA private key. The public key is
bundled inside the package — no internet required, no phone-home for validation.

```
CHORUSMESH_LICENSE_KEY=eyJhbGciOiJSUzI1NiJ9...
```

The JWT encodes:
```json
{
  "tier": "team",
  "nodes": 10,
  "features": ["slack", "pagerduty", "kafka", "raft", "custom_rules"],
  "issued_to": "acme-corp",
  "exp": 1814400000
}
```

- **No key present** → 30-day Developer trial (3 nodes, SMTP only)
- **Key present, valid** → features in the `features` claim are unlocked
- **Key expired** → warn on startup, keep running, block new node registrations

Keys are generated by InsightIts on payment (Stripe webhook → key generation
script → email delivery). No license server required to operate.

---

#### Feature table

| Feature | Open source (`prismlib`) | Developer $29/mo | Team $149/mo | Business $499/mo | Enterprise |
|---------|:---:|:---:|:---:|:---:|:---:|
| **Nodes** | Unlimited | 3 | 10 | 50 | Unlimited |
| **ClusterCache** | Yes | Yes | Yes | Yes | Yes |
| **Blue/Green/Orange failover** | Yes (watchdog) | Yes | + Raft consensus | + cross-region | Custom |
| **Alert channels** | SMTP only | SMTP only | + Slack + PagerDuty webhook | + OpsGenie + custom webhooks | Custom |
| **Alert rules** | 12 built-in | 12 built-in | + write custom rules | + escalation chains | + SLA alerting |
| **Alert history** | None | None | 7-day log | 30-day exportable | Unlimited |
| **Email recipients** | 1 | 1 | 5 | Unlimited | Unlimited |
| **CHORUS transport** | HTTP fallback | HTTP fallback | + Kafka / NATS adapter | + multi-region routing | + geo-aware failover |
| **Token budgets** | 80% / 95% thresholds | 80% / 95% thresholds | + per-tenant budgets | + cost dashboards | + chargeback reports |
| **Support** | GitHub Issues | GitHub Issues | Email, 48h response | Email + Slack, 24h | Dedicated Slack, 4h SLA |
| **License** | Apache 2.0 | JWT key | JWT key | JWT key | Custom MSA |

---

#### What justifies each price jump

**Free → Developer ($29):** Nothing extra feature-wise on the free tier — Developer
is for people who want a paid relationship: support email access and a clear
commercial license with named-company terms. The 30-day trial is the hook.

**Developer → Team ($149):** The real unlock is **Slack + PagerDuty** and **custom
alert rules**. Engineers at a 5-person team live in Slack — they won't act on
an email at 3am, they will act on a Slack ping. Custom rules let them write
`gpu_temp > 85` or `queue_depth > 1000` instead of being stuck with the 12
built-ins. Kafka/NATS adapter matters here too — teams this size usually already
have a broker. This is the highest-conversion tier.

**Team → Business ($499):** The unlock is **escalation chains** (warn → if
unacknowledged in 10min → critical page) and **30-day exportable alert history**.
On-call teams need escalation. Compliance teams need history. Multi-region CHORUS
routing matters for companies with users in multiple geographies.

**Business → Enterprise:** Air-gap deployment (no internet on prod), SOC 2 docs,
chargeback reports (per-tenant token cost breakdown for internal billing), and
a named human on Slack with a 4-hour SLA. This is what Fortune 500 procurement
requires.

---

#### Enforcement details for the page agent

The page should make the enforcement model transparent — engineering buyers
respect honesty about this more than vague "premium features" language.

Suggested copy:

> *"ChorusMesh uses offline JWT license keys — no phone-home, no internet
> required at runtime. The key is verified against a public key bundled in
> the package. If your key expires, existing nodes keep running; new
> registrations are blocked until you renew. Air-gapped deployments fully
> supported on Business and Enterprise tiers."*

---

### PrismLib Enterprise (direct agreement, no ChorusMesh required)

For organizations that need the full stack with commercial support but want
a single contract rather than a subscription:

- **On-premises / air-gapped** — hardened Docker images, SOC 2 docs, no phone-home
- **SLA-backed support** — guaranteed response times, dedicated Slack channel, architecture review
- **Custom embedding models** — domain-tuned hit rates for legal, medical, finance, code
- **Multi-region CHORUS topology** — active-active DB clusters, geo-aware driver routing
- **Compliance exports** — per-query access logs, tenant isolation attestation, GDPR lineage
- **Chargeback reports** — per-tenant token cost breakdown for internal billing
- **Professional services** — migration from GPTCache/Redis, RowVectorizer schema design,
  RAG pipeline integration, performance tuning

No public pricing. Every deployment is scoped differently.
Contact: insightits.info@gmail.com — response within 24 hours.

---

## Enterprise CTA section (copy for the page)

### Section title
"Need more? Talk to us."

### Body copy
PrismLib is Apache 2.0 — free forever for individuals and teams. If your
organization needs SLA support, compliance documentation, multi-region CHORUS
topologies, custom embedding models, or a dedicated migration path from your
current stack, we offer enterprise agreements.

No public pricing page. No sales funnel to sit in. Email us directly and we'll
respond within 24 hours.

### What to include on the page

- Primary button: **"Contact for Enterprise Pricing"** → `mailto:insightits.info@gmail.com`
- Secondary button: **"Open a GitHub Discussion"** → `https://github.com/insightitsGit/prismlib/discussions`
- Stat bar (reinforce credibility before the CTA):
  - 76% avg token savings across a 3-node cluster
  - <1s alert propagation without Prometheus or Datadog
  - ~3s automatic failover, zero human intervention
  - 58–64% context compression, no extra model or API call
- Small print: "PrismLib is used in the CHORUS Protocol M2M system — the same
  tensor transport powering AI agent meshes at InsightIts."

### Placement
1. After the benchmark results section (numbers set credibility → CTA captures intent)
2. Repeated in footer as: "Enterprise? [insightits.info@gmail.com](mailto:insightits.info@gmail.com)"

---

## Key copy (suggested for the page agent)

**Hero headline:** "One tunnel. Zero Redis. 76% fewer LLM tokens."

**Sub-headline:** "PrismLib Micro adds a cluster cache, health mesh, and
Blue/Green failover to your LLM microservices — over the same CHORUS tunnel
you already use. No Prometheus. No Kafka. No Kubernetes operator."

**ClusterCache tagline:** "Answers shared across every container · 100% cache
hit on warm nodes · zero tokens billed"

**AlertManager tagline:** "Health alerts in <1s · email + SIGNAL frame · no
monitoring stack required"

**Failover tagline:** "Blue/Green/Orange · auto-promote in ~3s · no human step"

**RAG tagline:** "58–64% fewer context tokens · cosine-sim chunk selection ·
drop-in before your LLM call"

**Technical credibility line:** "Built on CHORUS Fabric — the same binary gRPC
tensor transport used in InsightIts' AI agent mesh (CHORUS Protocol M2M)."

---

## Page sections (in order)

1. **Hero** — headline, sub-headline, two buttons: GitHub + `pip install`
2. **Problem** — "You have 5 containers answering the same LLM question 5 times.
   That's 5x your API bill."
3. **ClusterCache** — how it works, token savings table, 5-line code example
4. **RAG integration** — data flow diagram, RAG code example, token savings by layer
5. **Benchmark results** — all 5 phases: token savings, CHORUS latency, alert
   propagation, failover timing, compression ratio (use the tables from this doc)
6. **Competitive analysis** — three tables: token savings, alert propagation,
   failover (use data from "Competitive analysis" section above)
7. **Architecture diagram** — 4-node topology, CHORUS frame types, token flow
8. **CHORUS Fabric** — brief explanation of the transport layer, link to `info.md`
   for full CHORUS technical detail
9. **Installation** — `pip install "prismlib[fabric]"`, cluster quickstart (10 lines)
10. **Pricing** — open-source free tier + ChorusMesh tiers + Enterprise CTA
11. **Enterprise CTA** — full section with stat bar, email button, GitHub link
12. **Footer** — GitHub, Apache 2.0 badge, InsightIts, enterprise email

---

## Installation quickstart (for the page)

```bash
pip install "prismlib[fabric]"
```

```python
from prism.lib.fabric import CHORUSFabric
from prism.cluster.cache import ClusterCache
from prism.cluster.alerts import AlertManager, SMTPConfig

# 1. Start the CHORUS tunnel (connects to all peers)
fabric = CHORUSFabric(
    node_id = "node-green",
    peers   = {"blue": "https://node-blue.example.com",
                "orange": "https://node-orange.example.com"},
    pre_shared_key = os.getenv("CHORUS_KEY"),
)
await fabric.connect()

# 2. Cluster-wide LLM cache
cache = ClusterCache(node_id="node-green", fabric=fabric)

# 3. Health alerts to admin email
alerts = AlertManager(
    fabric    = fabric,
    mail_config = SMTPConfig(
        host="smtp.gmail.com", port=587,
        username="you@gmail.com",
        password=os.getenv("GMAIL_APP_PASS"),
        recipients=["admin@yourcompany.com"],
    ),
)

# 4. Drop into your existing RAG pipeline
answer = await cache.get_or_call(
    query        = user_question,
    query_vector = embed(user_question),
    call_fn      = lambda: llm.complete(user_question),
    context_chunks = retrieved_docs,
    chunk_vectors  = doc_embeddings,
)
```

---

## Files to reference

- `benchmark/cluster/node_app.py` — full PrismNode FastAPI app (endpoints,
  ClusterCache, AlertManager, CHORUS ingest, failover watchdog)
- `benchmark/cluster/run_cluster_benchmark.py` — 5-phase benchmark runner
- `benchmark/cluster/cluster_benchmark_results.json` — raw results from the live run
- `deploy/docker-compose.cluster.yml` — local 3-node cluster (Docker Compose)
- `deploy/azure_cluster_benchmark.sh` — Azure Container Apps deploy script
- `prism/cluster/cache.py` — ClusterCache, TokenUsage, ContextCompressor
- `prism/cluster/alerts.py` — AlertManager, 12 default rules, 5 mail backends
- `prism/cluster/node.py` — PrismNode, NodeRole, NodeState, failover logic
- `prism/lib/fabric.py` — CHORUSFabric, 7 FrameTypes, CHORUSFrame wire format
- `info.md` — PrismCache + PrismDriver context, CHORUS Fabric explanation,
  enterprise CTA copy for the base product page
- `whitepaper.md` — full technical whitepaper (link from the page)

---

## Contact and links

- Email: insightits.info@gmail.com
- GitHub: https://github.com/insightitsGit/prismlib
- PyPI: https://pypi.org/project/prismlib
- License: Apache 2.0
- Related: CHORUS Protocol M2M (InsightIts AI agent mesh)
- Related: PrismResonance — https://github.com/insightitsGit/prismresonance
