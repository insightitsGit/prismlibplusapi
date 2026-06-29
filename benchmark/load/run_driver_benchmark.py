"""
Driver benchmark — measures the latency difference between:

  Phase 1 (baseline): every query goes through the network to the DB node
  Phase 2 (driver):   every query hits the local in-process PrismResonance index

Usage:
  python benchmark/load/run_driver_benchmark.py \\
    --app-url  https://prism-benchmark.xxx.westus2.azurecontainerapps.io \\
    --db-url   https://prism-wrapper-sim.xxx.westus2.azurecontainerapps.io \\
    --users 30 --duration 60 --warmup-rows 5000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

SAMPLE_QUERIES = [
    "premium electronics item",
    "compact kitchen appliance",
    "deluxe sports equipment",
    "smart home device",
    "eco-friendly garden tool",
    "ultra lite clothing",
    "advanced automotive accessory",
    "classic book collection",
    "pro health supplement",
    "deluxe toy set for kids",
]


def wait_healthy(url: str, label: str, timeout: int = 60) -> bool:
    console.print(f"[yellow]Waiting for {label} at {url}/health ...[/yellow]")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                console.print(f"[green]✓ {label} healthy[/green]")
                return True
        except Exception:
            pass
        time.sleep(3)
    console.print(f"[red]✗ {label} not healthy at {url}[/red]")
    return False


def reset_driver(app_url: str) -> None:
    r = httpx.post(f"{app_url}/driver/reset", timeout=10)
    r.raise_for_status()
    d = r.json()
    console.print(f"[yellow]Driver index reset (evicted {d['evicted']} rows)[/yellow]")


def warmup_driver(app_url: str, rows: int) -> dict:
    """Wait for the subscription loop to fill the local index (no explicit warmup call needed)."""
    console.print(f"[cyan]Waiting for subscription loop to fill local index ({rows:,}+ rows) ...[/cyan]")
    t0 = time.monotonic()
    deadline = t0 + 120
    last_rows = 0
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{app_url}/driver/status", timeout=5)
            r.raise_for_status()
            status = r.json()
            received = status.get("rows_received", 0)
            is_warm  = status.get("is_warm", False)
            if received != last_rows:
                console.print(f"  rows_received={received} ...")
                last_rows = received
            if is_warm and received >= min(rows, 500):
                elapsed = time.monotonic() - t0
                throughput = received / elapsed if elapsed > 0 else 0
                console.print(
                    f"[green]✓ Index warm: {received:,} rows in {elapsed:.1f}s "
                    f"({throughput:.0f} rows/s)[/green]"
                )
                return {
                    "rows_loaded": received,
                    "index_size": status.get("index_size", received),
                    "elapsed_ms": elapsed * 1000,
                    "throughput_rows_per_s": throughput,
                }
        except Exception:
            pass
        time.sleep(3)
    console.print("[yellow]Warmup timeout — proceeding with available rows[/yellow]")
    status = httpx.get(f"{app_url}/driver/status", timeout=5).json()
    received = status.get("rows_received", 0)
    elapsed  = time.monotonic() - t0
    return {
        "rows_loaded": received,
        "index_size": status.get("index_size", received),
        "elapsed_ms": elapsed * 1000,
        "throughput_rows_per_s": received / elapsed if elapsed > 0 else 0,
    }


def run_locust_phase(
    app_url: str,
    endpoint: str,
    users: int,
    spawn_rate: int,
    duration: int,
    csv_prefix: str,
) -> Path:
    results_dir = Path("benchmark/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    locustfile = Path("benchmark/load/driver_locustfile.py")
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(locustfile),
        "--host", app_url,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", f"{duration}s",
        "--headless",
        "--csv", str(results_dir / csv_prefix),
        "--only-summary",
        "--loglevel", "WARNING",
    ]
    env_extra = {"DRIVER_ENDPOINT": endpoint}

    import os
    env = {**os.environ, **env_extra}

    console.print(f"[cyan]Locust: {users} users × {duration}s → {endpoint} ...[/cyan]")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        console.print(f"[red]Locust error:[/red]\n{proc.stderr[-1000:]}")

    return results_dir / f"{csv_prefix}_stats.csv"


def get_driver_metrics(app_url: str) -> dict:
    r = httpx.get(f"{app_url}/driver/metrics", timeout=10)
    r.raise_for_status()
    return r.json()


def print_report(
    baseline_metrics: dict,
    driver_metrics: dict,
    warmup_info: dict,
    args: argparse.Namespace,
) -> None:
    console.print()
    console.print(Panel.fit(
        "[bold]PrismDriver Benchmark — Baseline vs Local Index[/bold]",
        border_style="cyan",
    ))

    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("Metric", style="cyan", no_wrap=True)
    t.add_column("Baseline\n(DB node, network)", justify="right")
    t.add_column("Driver\n(local index)", justify="right")
    t.add_column("Improvement", justify="right")

    bm = baseline_metrics.get("baseline", {})
    dm = driver_metrics.get("driver", {})

    b_avg   = bm.get("avg_latency_ms", 0)
    d_avg   = dm.get("avg_latency_ms", 0)
    speedup = (b_avg / d_avg) if d_avg > 0 else 0.0
    pct     = ((b_avg - d_avg) / b_avg * 100) if b_avg > 0 else 0

    t.add_row(
        "Queries",
        f"{bm.get('queries', 0):,}",
        f"{dm.get('queries', 0):,}",
        "",
    )
    t.add_row(
        "Avg latency",
        f"[red]{b_avg:.1f}ms[/red]",
        f"[green]{d_avg:.3f}ms[/green]",
        f"[bold green]{speedup:.0f}×[/bold green]",
    )
    t.add_row(
        "Latency reduction",
        "",
        "",
        f"[bold green]{pct:.1f}%[/bold green]",
    )
    t.add_row(
        "Source",
        bm.get("source", ""),
        dm.get("source", ""),
        "",
    )
    t.add_row(
        "Index size (rows)",
        f"{warmup_info.get('index_size', 0):,}",
        f"{driver_metrics.get('index_size', 0):,}",
        "",
    )
    t.add_row(
        "Warmup time",
        f"{warmup_info.get('elapsed_ms', 0) / 1000:.1f}s",
        "",
        f"{warmup_info.get('throughput_rows_per_s', 0):.0f} rows/s",
    )

    console.print(t)
    console.print(
        f"\n[bold green]✓ PrismDriver is {speedup:.0f}× faster than direct DB access[/bold green]"
        f" — {pct:.1f}% latency reduction\n"
    )


def save_report(
    baseline_metrics: dict,
    driver_metrics: dict,
    warmup_info: dict,
    args: argparse.Namespace,
) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = Path("benchmark/results") / f"driver_benchmark_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp":        ts,
        "app_url":          args.app_url,
        "db_url":           args.db_url,
        "users":            args.users,
        "duration_s":       args.duration,
        "warmup_rows":      args.warmup_rows,
        "warmup_info":      warmup_info,
        "baseline_metrics": baseline_metrics,
        "driver_metrics":   driver_metrics,
        "speedup_factor":   driver_metrics.get("speedup_factor", 0),
    }
    path.write_text(json.dumps(report, indent=2))
    console.print(f"[dim]Report saved: {path}[/dim]")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="PrismDriver two-node benchmark")
    parser.add_argument("--app-url", required=True,
                        help="App node Container App URL")
    parser.add_argument("--db-url",  required=True,
                        help="DB node (wrapper-sim) Container App URL")
    parser.add_argument("--users",       type=int, default=30)
    parser.add_argument("--spawn-rate",  type=int, default=10)
    parser.add_argument("--duration",    type=int, default=60,
                        help="Seconds per phase")
    parser.add_argument("--warmup-rows", type=int, default=5000,
                        help="Rows to pull into local index before driver phase")
    args = parser.parse_args()

    console.print(f"[bold]PrismDriver Benchmark[/bold]")
    console.print(f"  App node : {args.app_url}")
    console.print(f"  DB node  : {args.db_url}")
    console.print(f"  Users    : {args.users} × {args.duration}s per phase\n")

    if not wait_healthy(args.app_url, "app node"):
        sys.exit(1)
    if not wait_healthy(args.db_url, "db node"):
        sys.exit(1)

    reset_driver(args.app_url)

    # ── Phase 1: Baseline (no local cache, network round-trip) ──
    console.print("\n[bold cyan]Phase 1 — Baseline (direct DB node queries)[/bold cyan]")
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_locust_phase(
        app_url=args.app_url,
        endpoint="/driver/baseline",
        users=args.users,
        spawn_rate=args.spawn_rate,
        duration=args.duration,
        csv_prefix=f"driver_baseline_{ts}",
    )
    baseline_metrics = get_driver_metrics(args.app_url)
    b_avg = baseline_metrics.get("baseline", {}).get("avg_latency_ms", 0)
    console.print(f"  Baseline avg latency: [red]{b_avg:.1f}ms[/red]")

    # ── Warmup: pull WAL rows into local index ──
    console.print("\n[bold cyan]Warmup — streaming WAL rows into local index[/bold cyan]")
    warmup_info = warmup_driver(args.app_url, args.warmup_rows)

    # ── Phase 2: Driver (local in-process index) ──
    console.print("\n[bold cyan]Phase 2 — Driver (local PrismResonance index)[/bold cyan]")
    run_locust_phase(
        app_url=args.app_url,
        endpoint="/driver/query",
        users=args.users,
        spawn_rate=args.spawn_rate,
        duration=args.duration,
        csv_prefix=f"driver_local_{ts}",
    )
    driver_metrics = get_driver_metrics(args.app_url)
    d_avg = driver_metrics.get("driver", {}).get("avg_latency_ms", 0)
    console.print(f"  Driver avg latency: [green]{d_avg:.3f}ms[/green]")

    print_report(baseline_metrics, driver_metrics, warmup_info, args)
    save_report(baseline_metrics, driver_metrics, warmup_info, args)


if __name__ == "__main__":
    main()
