# Release notes — prismlib-plus 0.8.0

**Release date:** 2026-07-18  
**PyPI package:** [`prismlib-plus`](https://pypi.org/project/prismlib-plus/)  
**Predecessor:** 0.7.0

## Highlights

PrismCache API parity for **PrismShine** coupling (aligned with prismlib 0.5.0):

- Selective invalidation: `invalidate_where(vector, threshold)`, `invalidate_tags(tags)`
- Tagged cache writes via `get_or_call(..., tags=[...])`
- Hit metadata: `HitMeta` / `last_hit_meta` + optional `on_hit` callback
- Eviction metrics in `CacheMetrics` and Prometheus (`evicted_by_vector` / `evicted_by_tags`)
- Docs fix: README uses `get_metrics()`; fabric/wrapper require `protobuf>=6.33.5`

Install:

```bash
pip install "prismlib-plus==0.8.0[enterprise,cache,fabric]"
```

See `CHANGELOG.md` § 0.8.0 for the full list.

---

# Release notes — prismlib-plus 0.7.0

**Release date:** 2026-06-29  
**PyPI package:** [`prismlib-plus`](https://pypi.org/project/prismlib-plus/) *(first publish of this superset)*  
**Predecessor on PyPI:** [`prismlib` 0.4.0](https://pypi.org/project/prismlib/) — cache, driver, and cluster core only

---

## What changed at a glance

PrismLib started as a **semantic LLM cache** and a **WAL-streaming DB driver**.  
**0.7.0** adds a full **agent API layer (PrismAPI)**, **enterprise security & observability**, and **Azure-validated end-to-end benchmarks** — without replacing the original stack. Everything still installs as optional extras.

| Before (prismlib 0.4.0) | New in 0.7.0 (prismlib-plus) |
|-------------------------|------------------------------|
| PrismCache — in-process LLM dedup | Same + OTel hooks, cache invalidation APIs |
| PrismDriver — WAL → local index | Same + gRPC Query/Write, HTTP subscribe, `is_warm` fix |
| PrismLib Micro — cluster cache / failover | Same + `ClusterTransport`, `HealthMonitor` |
| — | **PrismAPI** — vector-native provider/consumer over CHORUS frames |
| — | **Enterprise** — API keys, rate limits, audit log, Prometheus, mTLS |
| — | **MCP tool server** with optional API-key gate |
| — | Helm chart, Docker images, Azure 2-node driver e2e script |
| — | 217+ automated tests |

---

## Install

```bash
pip install "prismlib-plus[enterprise,cache,fabric]"
```

| Extra | What you get |
|-------|----------------|
| `[cache]` | PrismCache + embedders |
| `[fabric]` | PrismDriver, CHORUS client, cluster modules |
| `[wrapper]` | `prism-wrapper` daemon (DB node) |
| `[enterprise]` | PrismAPI FastAPI stack, auth, metrics |
| `[otel]` | OpenTelemetry tracing (optional) |
| `[all,enterprise]` | Full stack |

---

## Highlights

### 1. PrismAPI — vector-native API for AI agents

Providers embed content once; consumers receive **pre-projected float32 vectors** over CHORUS binary frames — no re-embedding on every retrieval.

```python
from prism.api import PrismAPIProvider, PrismAPIClient

# Provider embeds + projects results
@provider.expose
def search(query: str, top_k: int = 10) -> list[dict]:
    return db.search(query)[:top_k]

# Consumer gets vectors ready for retrieval
response = client.query("return policy", top_k=5)
vectors = response.vectors  # (N, dim) float32
```

Runnable examples: `examples/enterprise_server.py`, `examples/enterprise_client.py`, `examples/enterprise_golden_path.py`.

### 2. Enterprise-ready HTTP surface

- API key + bearer auth, per-client rate limiting
- Security audit log (`GET /audit/recent`)
- Prometheus metrics (`GET /metrics`)
- Optional mTLS on gRPC wrapper and driver client
- `SECURITY.md` threat model

### 3. PrismDriver — proven on Azure (2 containers)

Real load test on Azure Container Apps (westus2):

| Path | Avg read latency |
|------|------------------|
| Baseline (network to DB node) | **118.5 ms** |
| PrismDriver (local in-process index) | **0.27 ms** |

**439× faster · 99.8% latency reduction** (20 users × 45 s/phase)

Artifacts: `benchmark/results/driver_benchmark_azure.json`

> Benchmark uses `wrapper-sim` + Python driver mode. Production Postgres + C++ DLL will differ; numbers are real for the documented topology. See `BENCHMARK_RESULTS.md`.

### 4. PrismCache — still 91–96% hit rate under load

Azure concurrent Locust runs unchanged in quality; see `BENCHMARK_RESULTS.md` §1.

### 5. Cluster mesh — 76% token savings (3-node Azure)

Cross-VNet CHORUS mesh, failover, and compression benchmarks documented in `BENCHMARK_RESULTS.md` §3.

---

## Breaking changes

None for users of **prismlib 0.4.0** — this is a **new package name** (`prismlib-plus`).  
Import paths remain `prism.*`.

If you publish as **`prismlib` 0.5.0** instead (same name upgrade), document the new optional `[enterprise]` extra for PrismAPI users.

---

## Bug fixes

- **`LocalIndex.is_warm`** — correctly reports warm state after WAL ingest (fixed false 503s on `/driver/query` during benchmarks)
- **Benchmark reset** — `/driver/reset-baseline` clears baseline counters without evicting the warm index
- **Alert rules** — removed unsafe `eval` in condition parsing
- **TLS defaults** — gRPC paths require TLS or explicit `allow_insecure=True` (dev only)

---

## Migration from prismlib 0.4.0

```bash
# Was:
pip install "prismlib[cache,fabric]"

# Now (superset):
pip install "prismlib-plus[cache,fabric]"

# Add enterprise / PrismAPI:
pip install "prismlib-plus[enterprise,cache,fabric]"
```

Code imports unchanged: `from prism.cache import PrismCache`, `from prism.ffi import PrismDriver`, `from prism.api import PrismAPIProvider`.

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [README.md](README.md) | What PrismLib is, install, architecture |
| [RESULTS_AND_IMPROVEMENTS.md](RESULTS_AND_IMPROVEMENTS.md) | All benchmarks + shipped vs roadmap |
| [ENTERPRISE.md](ENTERPRISE.md) | Deploy guide (Docker, Helm, env vars) |
| [SECURITY.md](SECURITY.md) | Threat model |
| [RELEASE.md](RELEASE.md) | PyPI publish checklist |
| [CHANGELOG.md](CHANGELOG.md) | Full version history |

---

## Known limitations

- C++ `prism_driver` DLL — ABI header present; `.cpp` sources not in OSS repo (Python fallback used)
- Production wrapper requires real DB DSN; benchmarks use `wrapper-sim`
- Raft/consensus cluster HA — roadmap for paid tier

---

## Contributors & thanks

Built by [Insight IT Solutions](https://github.com/insightitsGit).  
Benchmark compute: Azure Container Apps westus2.

**License:** Apache 2.0
