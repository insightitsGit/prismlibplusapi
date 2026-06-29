# Results & improvements — single reference

**Package:** `prismlib-plus` 0.7.0 (superset of PyPI `prismlib` 0.4.0)  
**Last updated:** 2026-06-29

All benchmark numbers are from **real runs** (Azure Container Apps westus2 or documented local harness). Raw JSON under `benchmark/results/`.

---

## 1. Benchmark results (production-relevant)

### 1.1 PrismDriver — 2-container Azure e2e ✅ *latest*

| Metric | Baseline (network) | PrismDriver (local index) |
|--------|-------------------|---------------------------|
| Avg read latency | **118.5 ms** | **0.27 ms** |
| Speedup | — | **439×** |
| Latency reduction | — | **99.8%** |
| Load | 20 users × 45 s/phase | 4,886 driver queries |
| Driver mode | — | `python` (OSS; C++ DLL not in repo) |
| Index | 1,000-row catalog | WAL subscribe ~51k rows/s ingest |

- **Deploy:** `deploy/azure_driver_run.ps1`
- **Report:** `benchmark/results/driver_benchmark_azure.json`
- **Logs:** `benchmark/results/azure_e2e_logs/`
- **Honesty:** wrapper-sim, not live Postgres WAL; see `BENCHMARK_RESULTS.md` § Scope

### 1.2 PrismCache — Azure concurrent load

| Scenario | Hit rate | Tokens saved | Est. monthly $ saved |
|----------|----------|--------------|----------------------|
| Light (20 users, 60 s) | 91–92% | 953k | ~$412 |
| Mixed (50 users, 300 s) | **95.9%** | 1.67M | ~$723 |

### 1.3 Cluster mesh — Azure 3 nodes / 2 VNets

| Area | Result |
|------|--------|
| Token savings (cluster avg) | **76.1%** |
| CHORUS frame latency | 19–22 ms (same/cross VNet) |
| Failover detection | ~4 s, promotion ~97 ms |
| Context compression | **60.4%** avg token reduction |

---

## 2. Improvements shipped in 0.7.0

### Enterprise readiness

| Area | What was added |
|------|----------------|
| **PrismAPI** | Provider/consumer, CHORUS frames, `enterprise_server.py`, `enterprise_client.py` |
| **Auth** | API keys, bearer tokens, rate limiting, audit log |
| **TLS/mTLS** | Wrapper gRPC server + driver client; `scripts/gen_dev_certs.py` |
| **Observability** | Prometheus `/metrics`, optional OTel spans |
| **Packaging** | Helm chart, `Dockerfile.enterprise`, `ENTERPRISE.md`, `SECURITY.md` |
| **MCP** | Tool server with optional `api_key` gate |

### Driver & correctness

| Area | What was added |
|------|----------------|
| **LocalIndex** | WAL INSERT/UPDATE/DELETE by `row_id`; `is_warm` fix |
| **WAL path** | RowEventHub, publisher DELETE, HTTP + gRPC subscribe |
| **Python driver** | `grpc_client.py` Query/Write; subscription loop |
| **Benchmark** | 2-node Azure script, docker-compose driver, reset-baseline |

### Tests & CI

- 217+ pytest tests (enterprise, security, cluster, gRPC, MCP)
- Azure e2e reproducible via `deploy/azure_driver_run.ps1`

---

## 3. Improvements still open (roadmap)

From `IMPROVEMENTS.md` — not blockers for 0.7.0 publish:

| # | Item | Status |
|---|------|--------|
| 1 | Real HTTP PrismAPI benchmark at 128-dim | Open |
| 2 | InsightitsAIAgent production before/after case study | Open |
| 3 | BEIR validation for PrismResonance | Open |
| 4 | External user quote | Open |
| 5 | Federated search GTM (regulated industries) | Open |
| 6 | CHORUS licensing outreach | Open |
| 7 | Cryptographer review of CHORUS cipher | Open |
| 8 | Default 128-dim in docs/benchmarks | Open |

### Architecture gaps (documented, not fake)

- **C++ `prism_driver` DLL** — header + CMake only; Python fallback used in benchmarks
- **Production wrapper** — benchmark uses `wrapper-sim`; real `prism-wrapper` + DB DSN for enterprise pilot
- **Raft/consensus** — cluster HA is leaderless demo; paid-tier roadmap

---

## 4. What you can claim today vs enterprise SLA

| Claim | Supported by |
|-------|----------------|
| Local index >> remote vector query for read path | Azure 439× benchmark |
| Semantic cache 90%+ hit rate under load | Azure cache benchmarks |
| Cluster token sharing & failover demo | Azure 3-node run |
| Enterprise auth, metrics, mTLS scaffolding | Code + tests + ENTERPRISE.md |
| Production SLA (“always 0.27 ms”) | **Not yet** — needs real DB + soak test + APM |

---

## 5. Quick links

| Doc | Purpose |
|-----|---------|
| `BENCHMARK_RESULTS.md` | Full benchmark tables |
| `CHANGELOG.md` | Version history |
| `RELEASE.md` | PyPI publish steps |
| `ENTERPRISE.md` | Deploy guide |
| `SECURITY.md` | Threat model |
| `IMPROVEMENTS.md` | GTM / research roadmap |
