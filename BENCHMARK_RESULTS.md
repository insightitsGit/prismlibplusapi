# PrismLib — Benchmark Results (single source of truth)

All numbers below are from **real runs**. Two environments:

- **Cache & Driver layers** — Azure Container Apps (westus2), concurrent load test
- **Cluster mesh** — Azure Container Apps (westus2), 3 nodes across 2 VNets

Raw data:
- `benchmark/results/prism_mixed_*_report.json`, `prism_light_*_report.json` (cache)
- `benchmark/results/driver_benchmark_azure.json` (driver e2e, latest)
- `benchmark/results/driver_benchmark_*.json` (driver history)
- `benchmark/cluster/cluster_benchmark_results_azure.json` (cluster mesh)
- `benchmark/cluster/cluster_benchmark_results_loopback.json` (earlier loopback run, for reference)

---

## 1. PrismCache — semantic LLM cache (Azure, concurrent load)

| Scenario | Users | Duration | Queries | Hit rate | Tokens saved | Monthly est. saved |
|----------|-------|----------|---------|----------|--------------|--------------------|
| Light | 20 | 60 s | 4,050 | 91.0–92.0% | 953,344 | ~$412 |
| Mixed | 50 | 300 s | 6,973 | **95.9%** | 1,673,216 | ~$723 |

Avg hit latency ~291 ms, avg miss latency ~415 ms (mixed scenario).

## 2. PrismDriver — WAL-streamed DB driver (Azure, 2-container e2e)

**Deploy:** `deploy/azure_driver_run.ps1` (or `.sh`)  
**Topology:** `prism-wrapper-sim` (DB node) + `prism-benchmark` (app node with Python PrismDriver)  
**Latest run:** 2026-06-29 — `benchmark/results/driver_benchmark_azure.json`  
**Logs:** `benchmark/results/azure_e2e_logs/`

| Path | Avg read latency | Result |
|------|------------------|--------|
| Baseline (network to DB) | 118.5 ms | — |
| PrismDriver (local index, Python mode) | **0.27 ms** | **439× faster · 99.8% latency reduction** |

20 concurrent users × 45 s/phase, 1,000-row catalog, WAL subscribe warmup ~51k rows/s ingest.  
Driver mode: `python` (C++ DLL not built in OSS repo).

**Container URLs (rg-prism-driver-e2e, westus2):**
- App: `https://prism-benchmark.gentlesmoke-8bb70e10.westus2.azurecontainerapps.io`
- DB:  `https://prism-wrapper-sim.gentlesmoke-8bb70e10.westus2.azurecontainerapps.io`

Previous run (2026-06-24): 70.7× / 2.02 ms driver / 142.8 ms baseline — `driver_benchmark_20260624_135338.json`.

---

## 3. Cluster mesh — 3 nodes, Azure Container Apps, 2 VNets (westus2)

Topology: GREEN + BLUE in Environment A (one VNet), ORANGE in Environment B
(separate VNet), benchmark runner external/cross-network.
Deploy: `deploy/azure_cluster_run.sh`.

### 3.1 Token savings

| Node | Role | Network | Tokens billed | Tokens saved | Mechanism | Per-node savings |
|------|------|---------|---------------|--------------|-----------|------------------|
| node-green | active | same VNet | 328 | 130 | context compression | 28.4% |
| node-blue | warm standby | same VNet | 0 | 61 | cluster cache | 100% |
| node-orange | reserve | cross-VNet | 0 | 75 | cluster cache | 100% |
| **Cluster avg** | | | | | | **76.1%** |

> Note: 76.1% is the arithmetic mean of per-node savings under a workload where
> warm nodes receive already-answered queries. Conservative reading: savings on
> a node = (fraction of its queries already answered elsewhere) + 28–64%
> compression on the rest. High-repeat workloads trend toward the cache figure;
> low-repeat toward the compression-only figure.

### 3.2 CHORUS frame latency (cross-VNet, same region)

| Node | Network | Avg | Min | Max |
|------|---------|-----|-----|-----|
| node-green | same VNet | 19.4 ms | 15.0 ms | 29.0 ms |
| node-blue | same VNet | 19.9 ms | 15.2 ms | 22.6 ms |
| node-orange | cross-VNet | 21.6 ms | 16.0 ms | 26.3 ms |

### 3.3 Health-alert propagation

| Event | Source → Dest | Network | Propagation |
|-------|---------------|---------|-------------|
| cpu_high (92%) | GREEN → BLUE | same VNet | 633 ms |
| cpu_high (92%) | GREEN → ORANGE | cross-VNet | 674 ms |

### 3.4 Failover (leaderless)

| Metric | Value |
|--------|-------|
| GREEN silence threshold | 3,000 ms |
| Failover detected | 3,960 ms |
| Promotion to active (once detected) | 97 ms |
| Human intervention | none |

### 3.5 Context compression

| Query | Tokens used | Tokens saved | Compression |
|-------|-------------|--------------|-------------|
| What is PrismLib? | 71 | 126 | 64.0% |
| How does CHORUS Fabric work? | 85 | 118 | 58.1% |
| Explain context compression. | 75 | 128 | 63.1% |
| What is Blue/Green/Orange failover? | 84 | 116 | 58.0% |
| How does token deduplication work? | 81 | 116 | 58.9% |
| **Average** | | | **60.4%** |

---

## Scope & honesty notes

- Driver e2e uses **wrapper-sim** (in-memory catalog), not production Postgres + `prism-wrapper`. Numbers are real for that topology; validate on your DB before SLAs.
- Driver ran in **Python mode**; C++ DLL sources are not in the OSS repo.
- The cluster ran across two VNets but **both in westus2** — cross-*region*
  latency and network-partition behavior are untested.
- Cluster benchmark is a **functional 3-node run** (5 queries), not a sustained
  load test; the cache/driver layers (§1–2) are the load-tested ones.
- Leaderless promotion trades Raft-style consensus guarantees for simplicity;
  a partition could briefly produce two actives.
- Loopback-vs-Azure: token savings and compression were identical (logic, not
  network); only latency/alert/failover timings differ, and the Azure numbers
  are the canonical ones above.
