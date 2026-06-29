# Changelog

All notable changes to **prismlib-plus** (superset of [prismlib](https://pypi.org/project/prismlib/) on PyPI).

Format based on [Keep a Changelog](https://keepachangelog.com/).

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
