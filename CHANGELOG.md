# Changelog

All notable changes to **prismlib-plus** (superset of [prismlib](https://pypi.org/project/prismlib/) on PyPI).

Format based on [Keep a Changelog](https://keepachangelog.com/).

---

## [0.8.0] — 2026-07-18

### PrismCache — PrismShine coupling (API parity with prismlib 0.5.0)

- `invalidate_where(vector, threshold) -> int` — selective eviction by cosine similarity in tenant-projected space
- `invalidate_tags(tags) -> int` — evict entries matching any tag
- Optional `tags=` on `get_or_call` / `aget_or_call` (persisted; SQLite round-trip)
- `HitMeta` + `last_hit_meta` on hits (`created_at`, `tags`, `llm_model`, `score`) without changing return type
- Optional `on_hit` callback on `PrismCache.build(...)`
- Metrics: `CacheMetrics.evicted_by_vector`, `evicted_by_tags`
- Observability: `prism_cache_evicted_by_vector_total`, `prism_cache_evicted_by_tags_total`
- Persist projected `query_vector` on store entries so vector invalidation works after cold restart

### Fixes

- README: `cache.metrics()` → `get_metrics()` (and correct metric field names)
- Proto stub import: relative `from . import chorus_pb2` for package installs
- Raise fabric/wrapper `protobuf` floor to `>=6.33.5` (matches checked-in gencode)

---

## [0.7.0] — 2026-06-29

### Azure-validated benchmarks

- **PrismDriver 2-container e2e** (westus2): **118.5 ms → 0.27 ms** (439×, 99.8% latency reduction)
  - Deploy: `deploy/azure_driver_run.ps1` / `.sh`
  - Artifacts: `benchmark/results/driver_benchmark_azure.json`, `benchmark/results/azure_e2e_logs/`
- **PrismCache** (Azure): 91–96% hit rate under concurrent load
- **Cluster mesh** (Azure, 3 nodes / 2 VNets): 76% token savings, ~20 ms CHORUS latency, ~4 s failover

### Enterprise & security

- `PrismAPI` enterprise FastAPI stack: auth, rate limit, audit, `/health`, `/metrics`
- mTLS on `WrapperGrpcServer` and `PrismDriver` gRPC client
- `SECURITY.md` threat model; safe alert rules (no `eval`)
- MCP server API-key gate (`PRISM_MCP_API_KEY`)
- Helm chart with API-key secrets, TLS/mTLS projected volumes
- `ENTERPRISE.md`, `examples/enterprise_server.py`, `enterprise_client.py`, Docker/compose

### Correctness & driver

- `LocalIndex` WAL upsert/update/delete by `row_id`
- `RowEventHub`, HTTP subscribe server, gRPC `WrapperService` (Query/Write/Subscribe)
- `prism.ffi.grpc_client` — remote Query/Write for Python driver
- `PrismDriver` gRPC subscription + HTTP fallback (`PRISM_WRAPPER_URL`)
- Fix: `LocalIndex.is_warm` now reflects ingested rows (was stale `_rows`)
- Benchmark: `/driver/reset-baseline` preserves warm index between phases

### Observability

- Prometheus metrics registry; optional OpenTelemetry spans on cache + API client
- `ClusterTransport`, `HealthMonitor`, cluster cache invalidation hooks

### Tests

- 217+ tests: enterprise, security, cluster, gRPC client, MCP auth

---

## [0.6.0] — 2026-06 (internal)

- Initial enterprise tier implementation (auth, gRPC wrapper, observability scaffolding)
- Version bump in repo; not published to PyPI

---

## PyPI lineage

| Package | PyPI version | Source |
|---------|--------------|--------|
| `prismlib` | **0.4.0** (latest on PyPI) | [PrismLib](https://github.com/insightitsGit/prismlib) — cache, driver, cluster core |
| `prismlib-plus` | **0.7.0** (this release) | PrismLabPlusAPI — prismlib + PrismAPI + enterprise |

Install the superset:

```bash
pip install "prismlib-plus[enterprise,cache,fabric]"
```

To publish as an upgrade to the existing package name, see `RELEASE.md` (optional `prismlib` 0.5.0 rename).

**User-facing release announcement:** [RELEASE_NOTES.md](RELEASE_NOTES.md)
