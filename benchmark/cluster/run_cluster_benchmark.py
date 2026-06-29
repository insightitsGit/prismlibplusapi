"""
4-Container Cluster Benchmark — PrismLib ClusterCache + CHORUS Fabric
======================================================================

Topology:
  GREEN  (node-1) ─┐  same Container App Environment (same VNet)
  BLUE   (node-2) ─┘
  ORANGE (node-3)    separate Container App Environment (cross-network)
  RUNNER (this script) — talks to all three externally

Metrics measured:
  1. Token savings across pipeline layers (local / cluster / compression / dedup)
  2. CHORUS frame latency — same-pod vs cross-network
  3. Health alert propagation time
  4. Failover time (GREEN pause → BLUE promotion)
  5. Context compression ratio

Usage:
  python run_cluster_benchmark.py \
    --green  https://node-green.xyz.azurecontainerapps.io \
    --blue   https://node-blue.xyz.azurecontainerapps.io \
    --orange https://node-orange.xyz.azurecontainerapps.io
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def get(client: httpx.AsyncClient, url: str, path: str) -> dict:
    r = await client.get(f"{url}{path}", timeout=10.0)
    r.raise_for_status()
    return r.json()


async def post(client: httpx.AsyncClient, url: str, path: str, body: dict = {}) -> dict:
    r = await client.post(f"{url}{path}", json=body, timeout=10.0)
    r.raise_for_status()
    return r.json()


async def wait_ready(client: httpx.AsyncClient, url: str, label: str, max_wait: int = 120) -> bool:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = await client.get(f"{url}/health", timeout=5.0)
            if r.status_code == 200:
                console.log(f"[green]OK[/] {label} is ready")
                return True
        except Exception:
            pass
        await asyncio.sleep(2)
    console.log(f"[red]FAIL[/] {label} not ready after {max_wait}s")
    return False


# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

QUERIES = [
    ("What is PrismLib?", ["PrismLib is a tensor-native LLM cache.",
                            "It was designed for multi-tenant isolation.",
                            "Context compression reduces token usage by 90%.",
                            "PrismCache uses JL projection for fast similarity.",
                            "PrismDriver streams WAL events via CHORUS Fabric."]),
    ("How does CHORUS Fabric work?", ["CHORUS uses encrypted gRPC binary frames.",
                                       "TensorCipher applies V_enc = V @ K.",
                                       "HMAC-SHA256 watermarks verify integrity.",
                                       "Ephemeral key rotation happens every 1000 frames.",
                                       "7 frame types: VECTOR, DELTA, SIGNAL, CONFIG, METRIC, HEALTH, APP_EVENT."]),
    ("Explain context compression.", ["Top-K cosine similarity keeps relevant chunks.",
                                       "Irrelevant chunks are dropped before LLM call.",
                                       "This reduces prompt tokens by up to 90%.",
                                       "Hot chunks are learned from METRIC frames.",
                                       "Compression ratio improves over time."]),
    ("What is Blue/Green/Orange failover?", ["GREEN is the active master.",
                                              "BLUE is a warm standby synced via CHORUS.",
                                              "ORANGE is the reserve reserve node.",
                                              "Failover time is approximately 50ms.",
                                              "BLUE watches GREEN heartbeats every 1s."]),
    ("How does token deduplication work?", ["In-flight queries are deduplicated via asyncio Future.",
                                             "Identical concurrent queries share one LLM call.",
                                             "The result is broadcast to all waiting callers.",
                                             "Token savings scale with query concurrency.",
                                             "Zero extra latency for deduplicated callers."]),
    # Repeat first 2 to test cache hits
    ("What is PrismLib?", []),
    ("How does CHORUS Fabric work?", []),
]

CONTEXT_CHUNKS = [
    "PrismLib is a tensor-native semantic cache for LLM applications.",
    "CHORUS Fabric provides encrypted binary streaming via gRPC.",
    "ClusterCache shares token savings across all nodes in the cluster.",
    "PrismResonance uses wave-interference for sub-millisecond similarity.",
    "Blue/Green/Orange topology ensures zero-downtime failover.",
    "Context compression uses cosine similarity to keep only top-K chunks.",
    "JL projection seeded by SHA-256(tenant_id) isolates tenants mathematically.",
    "AlertManager fires emails when CPU, RAM, or latency thresholds are crossed.",
    "Token deduplication coalesces identical in-flight LLM calls into one.",
    "The 98.6% latency reduction was proven on two-node Azure Container Apps.",
]


# ---------------------------------------------------------------------------
# Benchmark phases
# ---------------------------------------------------------------------------

async def phase_reset(client: httpx.AsyncClient, nodes: dict[str, str]) -> None:
    for role, url in nodes.items():
        try:
            await post(client, url, "/admin/reset")
            console.log(f"[dim]Reset {role}[/]")
        except Exception as e:
            console.log(f"[yellow]Reset {role} failed: {e}[/]")


async def phase_token_savings(client: httpx.AsyncClient, nodes: dict[str, str]) -> dict:
    """Send 7 queries to GREEN, then same queries to BLUE and ORANGE to measure cache propagation."""
    console.rule("[bold cyan]Phase 1 — Token Savings Pipeline")
    results = []

    for i, (query, chunks) in enumerate(QUERIES, 1):
        role = "green" if i <= 5 else ("blue" if i == 6 else "orange")
        target_url = nodes.get(role, nodes["green"])

        body = {
            "query": query,
            "context_chunks": chunks or CONTEXT_CHUNKS[:6],
            "use_cluster_cache": True,
            "use_compression": True,
        }
        t0 = time.monotonic()
        try:
            r = await post(client, target_url, "/query", body)
            latency = (time.monotonic() - t0) * 1000
            results.append({
                "query":          query[:50],
                "target":         role,
                "source":         r.get("source", "?"),
                "tokens_used":    r.get("tokens_used", 0),
                "tokens_saved":   r.get("tokens_saved", 0),
                "compressed":     r.get("tokens_compressed", 0),
                "latency_ms":     round(latency, 1),
                "savings_type":   r.get("savings_type", "?"),
            })
            source_color = {
                "llm_call":     "red",
                "cluster_cache": "green",
                "dedup":        "blue",
            }.get(r.get("source",""), "yellow")
            console.log(f"  Q{i} [{role}] → [{source_color}]{r.get('source','?')}[/] "
                        f"tokens={r.get('tokens_used',0)} saved={r.get('tokens_saved',0)} "
                        f"lat={latency:.0f}ms")
        except Exception as e:
            console.log(f"  [red]Q{i} failed: {e}[/]")

    await asyncio.sleep(1)   # let TOKEN_SYNC frames propagate

    # Collect totals from all nodes
    totals = {}
    for role, url in nodes.items():
        try:
            m = await get(client, url, "/metrics/tokens")
            totals[role] = m
        except Exception as e:
            console.log(f"[yellow]Metrics from {role} failed: {e}[/]")

    return {"queries": results, "node_totals": totals}


async def phase_chorus_latency(client: httpx.AsyncClient, nodes: dict[str, str]) -> dict:
    """Send HEALTH frames and measure transit time same-pod vs cross-network."""
    console.rule("[bold cyan]Phase 2 — CHORUS Frame Latency")
    results = {}

    for role, url in nodes.items():
        try:
            m = await get(client, url, "/metrics/chorus")
            results[role] = m
            console.log(f"  {role}: avg={m.get('avg_latency_ms',0):.1f}ms "
                        f"p99={m.get('p99_latency_ms',0):.1f}ms "
                        f"sent={m.get('frames_sent',0)} recv={m.get('frames_received',0)}")
        except Exception as e:
            console.log(f"  [yellow]{role} chorus metrics failed: {e}[/]")

    return results


async def phase_alert_propagation(client: httpx.AsyncClient, nodes: dict[str, str]) -> dict:
    """Trigger an alert on GREEN, measure how fast BLUE and ORANGE see it."""
    console.rule("[bold cyan]Phase 3 — Alert Propagation")

    # Record baseline alert count on all nodes
    baseline = {}
    for role, url in nodes.items():
        try:
            a = await get(client, url, "/alerts")
            baseline[role] = len(a.get("alerts", []))
        except Exception:
            baseline[role] = 0

    # Trigger a real alert on GREEN — it will broadcast SIGNAL frames to BLUE + ORANGE
    t_fire = time.time()
    try:
        await post(client, nodes["green"], "/admin/fire_test_alert")
        console.log("  [yellow]Alert fired on GREEN — broadcasting to peers[/]")
    except Exception as e:
        console.log(f"  [red]Alert fire failed: {e}[/]")

    # Poll until BLUE and ORANGE see the alert (max 10s)
    propagation: dict[str, Optional[float]] = {"blue": None, "orange": None}
    deadline = time.time() + 10
    while time.time() < deadline:
        await asyncio.sleep(0.5)
        for role in ("blue", "orange"):
            if propagation[role] is not None:
                continue
            try:
                a = await get(client, nodes[role], "/alerts")
                count = len(a.get("alerts", []))
                if count > baseline.get(role, 0):
                    propagation[role] = round((time.time() - t_fire) * 1000, 1)
                    console.log(f"  [green]Alert seen on {role} in {propagation[role]:.0f}ms[/]")
            except Exception:
                pass
        if all(v is not None for v in propagation.values()):
            break

    for role, ms in propagation.items():
        if ms is None:
            console.log(f"  [red]Alert NOT propagated to {role} within 10s[/]")

    return {"alert_propagation_ms": propagation, "fired_at": t_fire}


async def phase_failover(client: httpx.AsyncClient, nodes: dict[str, str]) -> dict:
    """Pause GREEN heartbeats, measure time until BLUE promotes itself."""
    console.rule("[bold cyan]Phase 4 — Failover Simulation")

    t_start = time.time()
    try:
        await post(client, nodes["green"], "/admin/simulate_failover")
        console.log("  [yellow]GREEN heartbeat paused for 5s[/]")
    except Exception as e:
        console.log(f"  [red]simulate_failover failed: {e}[/]")
        return {"error": str(e)}

    # Poll BLUE until its role flips to green
    failover_ms: Optional[float] = None
    deadline = time.time() + 15
    while time.time() < deadline:
        await asyncio.sleep(0.2)
        try:
            s = await get(client, nodes["blue"], "/status")
            if s.get("role") == "green":
                failover_ms = round((time.time() - t_start) * 1000, 1)
                console.log(f"  [green bold]BLUE promoted to GREEN in {failover_ms:.0f}ms[/]")
                break
        except Exception:
            pass

    fm = await get(client, nodes["blue"], "/metrics/failover")
    return {
        "failover_detected_ms": failover_ms,
        "failover_duration_ms": fm.get("failover_duration_ms"),
        "failovers_triggered":  fm.get("failovers_triggered"),
        "new_green":            nodes["blue"],
    }


async def phase_compression_ratio(client: httpx.AsyncClient, nodes: dict[str, str]) -> dict:
    """Send 5 queries with long context, measure compression savings."""
    console.rule("[bold cyan]Phase 5 — Context Compression Ratio")
    results = []

    for i, (q, _) in enumerate(QUERIES[:5]):
        body = {
            "query":           q,
            "context_chunks":  CONTEXT_CHUNKS,  # all 10 chunks
            "use_cluster_cache": False,
            "use_compression": True,
        }
        try:
            r = await post(client, nodes["green"], "/query", body)
            compressed = r.get("tokens_compressed", 0)
            used       = r.get("tokens_used", 0)
            total      = used + compressed
            ratio      = round(compressed / total * 100, 1) if total > 0 else 0
            results.append({
                "query": q[:50], "tokens_used": used,
                "tokens_compressed": compressed,
                "compression_pct": ratio,
            })
            console.log(f"  Q{i+1}: saved {compressed} tokens via compression ({ratio}%)")
        except Exception as e:
            console.log(f"  [red]Q{i+1} failed: {e}[/]")

    return {"queries": results}


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(results: dict, nodes: dict[str, str]) -> None:
    console.print()
    console.print(Panel.fit(
        "[bold white]PrismLib 4-Container Cluster Benchmark Results[/]\n"
        "[dim]CHORUS Fabric · ClusterCache · AlertManager · Blue/Green/Orange Failover[/]",
        border_style="bright_cyan",
    ))

    # ── Token savings ─────────────────────────────────────────────────────
    console.print()
    t = Table("Node", "Role / Network", "Billed", "Cluster Hit", "Compressed",
              "Deduped", "Total Saved", "Savings %",
              title="[bold]Token Savings by Node[/]", box=box.ROUNDED)

    node_totals: dict = results.get("tokens", {}).get("node_totals", {})
    for role, m in node_totals.items():
        saved = (m.get("tokens_cached_local",0) + m.get("tokens_cached_cluster",0) +
                 m.get("tokens_compressed",0)   + m.get("tokens_deduped",0))
        network = {"green": "same-pod", "blue": "same-pod", "orange": "cross-network"}.get(role, "?")
        t.add_row(
            m.get("node_id", role),
            f"{role.upper()} / {network}",
            str(m.get("tokens_billed", 0)),
            str(m.get("tokens_cached_cluster", 0)),
            str(m.get("tokens_compressed", 0)),
            str(m.get("tokens_deduped", 0)),
            f"[green]{saved}[/]",
            f"[bold green]{m.get('savings_pct', 0):.1f}%[/]",
        )
    console.print(t)

    # ── CHORUS latency ────────────────────────────────────────────────────
    console.print()
    t2 = Table("Node", "Network Type", "Avg (ms)", "Min (ms)", "Max (ms)", "P99 (ms)",
               "Frames Sent", "Frames Recv",
               title="[bold]CHORUS Frame Latency[/]", box=box.ROUNDED)
    chorus: dict = results.get("chorus", {})
    network_type = {"green": "same-pod", "blue": "same-pod", "orange": "cross-network"}
    for role, m in chorus.items():
        t2.add_row(
            m.get("node_id", role), network_type.get(role, "?"),
            f"[cyan]{m.get('avg_latency_ms',0):.1f}[/]",
            str(m.get("min_latency_ms", 0)),
            str(m.get("max_latency_ms", 0)),
            f"[yellow]{m.get('p99_latency_ms',0):.1f}[/]",
            str(m.get("frames_sent", 0)),
            str(m.get("frames_received", 0)),
        )
    console.print(t2)

    # ── Alert propagation ─────────────────────────────────────────────────
    console.print()
    ap = results.get("alerts", {}).get("alert_propagation_ms", {})
    t3 = Table("Destination", "Network", "Propagation (ms)",
               title="[bold]Alert Propagation Time[/]", box=box.ROUNDED)
    for role, ms in ap.items():
        net = network_type.get(role, "?")
        t3.add_row(role.upper(), net,
                   f"[green]{ms:.0f}[/]" if ms else "[red]TIMEOUT[/]")
    console.print(t3)

    # ── Failover ──────────────────────────────────────────────────────────
    console.print()
    fo = results.get("failover", {})
    t4 = Table("Metric", "Value", title="[bold]Failover Timing[/]", box=box.ROUNDED)
    t4.add_row("GREEN silence threshold", "3 000 ms")
    t4.add_row("Failover detected in",
               f"[bold green]{fo.get('failover_detected_ms','?'):.0f} ms[/]"
               if fo.get("failover_detected_ms") else "[red]n/a[/]")
    t4.add_row("Promotion duration",
               f"{fo.get('failover_duration_ms','?'):.1f} ms"
               if fo.get("failover_duration_ms") else "n/a")
    t4.add_row("Failovers triggered", str(fo.get("failovers_triggered", 0)))
    t4.add_row("New GREEN node", fo.get("new_green", "?"))
    console.print(t4)

    # ── Compression ratio ─────────────────────────────────────────────────
    console.print()
    cr = results.get("compression", {}).get("queries", [])
    t5 = Table("Query", "Tokens Used", "Tokens Saved", "Compression %",
               title="[bold]Context Compression Ratio[/]", box=box.ROUNDED)
    for row in cr:
        t5.add_row(row["query"], str(row["tokens_used"]),
                   str(row["tokens_compressed"]),
                   f"[green]{row['compression_pct']}%[/]")
    console.print(t5)

    # ── Summary ───────────────────────────────────────────────────────────
    all_savings = [m.get("savings_pct", 0) for m in node_totals.values()]
    avg_savings = sum(all_savings) / max(len(all_savings), 1)

    same_pod_ms  = [m.get("avg_latency_ms",0) for r, m in chorus.items()
                    if r in ("green","blue")]
    cross_net_ms = [m.get("avg_latency_ms",0) for r, m in chorus.items()
                    if r == "orange"]

    console.print()
    console.print(Panel(
        f"  [bold]Avg token savings across cluster:[/] [green bold]{avg_savings:.1f}%[/]\n"
        f"  [bold]CHORUS latency — same-pod:[/]        "
        f"[cyan]{sum(same_pod_ms)/max(len(same_pod_ms),1):.1f} ms avg[/]\n"
        f"  [bold]CHORUS latency — cross-network:[/]   "
        f"[yellow]{sum(cross_net_ms)/max(len(cross_net_ms),1):.1f} ms avg[/]\n"
        f"  [bold]Alert propagation (same-pod):[/]     "
        f"[green]{ap.get('blue','?')} ms[/]\n"
        f"  [bold]Alert propagation (cross-net):[/]    "
        f"[yellow]{ap.get('orange','?')} ms[/]\n"
        f"  [bold]Failover time:[/]                    "
        f"[green bold]{fo.get('failover_detected_ms','?')} ms[/]",
        title="[bold white]Summary[/]",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    nodes = {
        "green":  args.green,
        "blue":   args.blue,
        "orange": args.orange,
    }

    console.print(Panel.fit(
        "[bold cyan]PrismLib 4-Container Cluster Benchmark[/]\n\n"
        f"  GREEN  (same-pod):     {args.green}\n"
        f"  BLUE   (same-pod):     {args.blue}\n"
        f"  ORANGE (cross-net):    {args.orange}\n",
        border_style="cyan",
    ))

    async with httpx.AsyncClient() as client:
        # Wait for all nodes
        console.rule("Waiting for nodes to be ready")
        ready = await asyncio.gather(
            wait_ready(client, nodes["green"],  "GREEN  (same-pod)"),
            wait_ready(client, nodes["blue"],   "BLUE   (same-pod)"),
            wait_ready(client, nodes["orange"], "ORANGE (cross-net)"),
        )
        if not all(ready):
            console.print("[red bold]Some nodes not ready — aborting.[/]")
            return

        await asyncio.sleep(3)  # let heartbeat frames exchange

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      TimeElapsedColumn(), console=console) as progress:

            t = progress.add_task("Resetting node stats...", total=None)
            await phase_reset(client, nodes)
            progress.remove_task(t)

        # Run all 5 phases
        token_data      = await phase_token_savings(client, nodes)
        await asyncio.sleep(5)   # allow cross-node CHORUS frames to settle
        chorus_data     = await phase_chorus_latency(client, nodes)
        alert_data      = await phase_alert_propagation(client, nodes)
        failover_data   = await phase_failover(client, nodes)
        compression_data = await phase_compression_ratio(client, nodes)

    results = {
        "tokens":      token_data,
        "chorus":      chorus_data,
        "alerts":      alert_data,
        "failover":    failover_data,
        "compression": compression_data,
    }

    # Save JSON
    import pathlib
    out = pathlib.Path(__file__).parent / "cluster_benchmark_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    console.print(f"\n[dim]Raw results saved to {out}[/]")

    render_report(results, nodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PrismLib 4-Container Cluster Benchmark")
    parser.add_argument("--green",  required=True, help="GREEN node URL")
    parser.add_argument("--blue",   required=True, help="BLUE node URL")
    parser.add_argument("--orange", required=True, help="ORANGE node URL")
    asyncio.run(main(parser.parse_args()))
