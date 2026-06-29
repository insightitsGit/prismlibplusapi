# PrismLib — Benchmark Results (single source of truth)

All numbers below are from **real runs**. Two environments:

- **Cache & Driver layers** — Azure Container Apps (westus2), concurrent load test
- **Cluster mesh** — Azure Container Apps (westus2), 3 nodes across 2 VNets

Raw data:
- `benchmark/results/prism_mixed_*_report.json`, `prism_light_*_report.json` (cache)
- `benchmark/results/driver_benchmark_*.json` (driver)
- `benchmark/cluster/cluster_benchmark_results_azure.json` (cluster mesh)
- `benchmark/cluster/cluster_benchmark_results_loopback.json` (earlier loopback run, for reference)

---

## 1. PrismCache — semantic LLM cache (Azure, concurrent load)

| Scenario | Users | Duration | Queries | Hit rate | Tokens saved | Monthly est. saved |
|----------|-------|----------|---------|----------|--------------|--------------------|
| Light | 20 | 60 s | 4,050 | 91.0–92.0% | 953,344 | ~$412 |
| Mixed | 50 | 300 s | 6,973 | **95.9%** | 1,673,216 | ~$723 |

Avg hit latency ~291 ms, avg miss latency ~415 ms (mixed scenario).

## 2. PrismDriver — WAL-streamed DB driver (Azure)

| Path | Avg read latency | Result |
|------|------------------|--------|
| Baseline (network to DB) | 142.83 ms | — |
| PrismDriver (local index) | **2.02 ms** | **70.7× faster · 98.6% latency reduction** |

30 concurrent users × 60 s/phase, 11,000-row index, warmup throughput ~26k rows/s.

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

- The cluster ran across two VNets but **both in westus2** — cross-*region*
  latency and network-partition behavior are untested.
- Cluster benchmark is a **functional 3-node run** (5 queries), not a sustained
  load test; the cache/driver layers (§1–2) are the load-tested ones.
- Leaderless promotion trades Raft-style consensus guarantees for simplicity;
  a partition could briefly produce two actives.
- Loopback-vs-Azure: token savings and compression were identical (logic, not
  network); only latency/alert/failover timings differ, and the Azure numbers
  are the canonical ones above.
