"""
Benchmark runner — seeds the cache, triggers Locust load, and prints
a final report with all key metrics.

Usage:
  # Local (no Azure, mock LLM):
  python benchmark/load/run_benchmark.py --scenario mixed --duration 60

  # Azure:
  python benchmark/load/run_benchmark.py \
    --host https://prism-benchmark.azurecontainerapps.io \
    --scenario mixed --duration 300 --users 100

  # CI smoke test:
  python benchmark/load/run_benchmark.py --scenario smoke --duration 10 --no-azure
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
from rich.progress import track

console = Console()


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

SCENARIOS = {
    "smoke":   {"users": 5,   "spawn_rate": 2,  "duration": 10,  "seed": 50},
    "light":   {"users": 20,  "spawn_rate": 5,  "duration": 60,  "seed": 500},
    "mixed":   {"users": 50,  "spawn_rate": 10, "duration": 300, "seed": 2000},
    "heavy":   {"users": 200, "spawn_rate": 20, "duration": 600, "seed": 10000},
    "extreme": {"users": 500, "spawn_rate": 50, "duration": 900, "seed": 50000},
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def wait_healthy(host: str, timeout: int = 60) -> bool:
    console.print(f"[yellow]Waiting for app at {host}/health ...[/yellow]")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{host}/health", timeout=3)
            if r.status_code == 200:
                console.print(f"[green]✓ App is healthy[/green]")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def seed_cache(host: str, count: int) -> None:
    console.print(f"[cyan]Seeding cache with {count:,} Q&A pairs ...[/cyan]")
    t0 = time.monotonic()
    r = httpx.post(f"{host}/admin/seed", params={"count": count}, timeout=300)
    r.raise_for_status()
    elapsed = time.monotonic() - t0
    data = r.json()
    console.print(
        f"[green]✓ Seeded {data['seeded']:,} entries in {elapsed:.1f}s "
        f"(cache_size={data['cache_size']})[/green]"
    )


def reset_cache(host: str) -> None:
    r = httpx.post(f"{host}/admin/reset", timeout=30)
    r.raise_for_status()
    console.print(f"[yellow]Cache reset ({r.json()['evicted']} evicted)[/yellow]")


def get_metrics(host: str) -> dict:
    r = httpx.get(f"{host}/metrics", timeout=10)
    r.raise_for_status()
    return r.json()


def run_locust(
    host: str,
    users: int,
    spawn_rate: int,
    duration: int,
    csv_prefix: str,
    no_azure: bool,
) -> Path:
    results_dir = Path("benchmark/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "locust",
        "-f", "benchmark/load/locustfile.py",
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", f"{duration}s",
        "--headless",
        "--csv", str(results_dir / csv_prefix),
        "--only-summary",
    ]

    if no_azure:
        # Use only WarmUser and MixedUser in smoke mode
        cmd += ["--class-picker"]

    console.print(f"[cyan]Running Locust: {users} users × {duration}s ...[/cyan]")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        console.print(f"[red]Locust error:[/red]\n{proc.stderr[-2000:]}")
        sys.exit(1)

    return results_dir / f"{csv_prefix}_stats.csv"


def print_report(metrics_before: dict, metrics_after: dict, csv_path: Path) -> None:
    console.print("\n[bold]── PrismCache Benchmark Report ────────────────────────────[/bold]")

    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("Metric", style="cyan", no_wrap=True)
    t.add_column("Value", justify="right")

    m = metrics_after
    t.add_row("Total queries",         f"{m['total_queries']:,}")
    t.add_row("Cache hits",            f"{m['total_hits']:,}")
    t.add_row("Cache misses",          f"{m['total_misses']:,}")
    t.add_row("Hit rate",              f"[green]{m['hit_rate_pct']:.1f}%[/green]")
    t.add_row("Avg hit latency",       f"[green]{m['avg_hit_latency_ms']:.1f}ms[/green]")
    t.add_row("Avg miss latency",      f"{m['avg_miss_latency_ms']:.0f}ms")
    t.add_row("Speedup factor",        f"[bold green]{m['speedup_factor']:.0f}×[/bold green]")
    t.add_row("Tokens saved",          f"{m['total_tokens_saved']:,}")
    t.add_row("Cost saved (session)",  f"${m['total_cost_saved_usd']:.4f}")
    t.add_row("Projected monthly",     f"[bold]${m['projected_monthly_usd']:,.2f}[/bold]")
    t.add_row("Cache size (entries)",  f"{m['cache_size']:,}")

    console.print(t)

    if csv_path.exists():
        console.print(f"\n[dim]Locust CSV: {csv_path}[/dim]")

    console.print("[bold green]✓ Benchmark complete[/bold green]\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PrismCache benchmark runner")
    parser.add_argument("--host",     default="http://localhost:8000")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="mixed")
    parser.add_argument("--duration", type=int, default=0,
                        help="Override scenario duration (seconds)")
    parser.add_argument("--users",    type=int, default=0,
                        help="Override concurrent users")
    parser.add_argument("--no-azure", action="store_true",
                        help="Skip Azure-specific setup, run locally")
    parser.add_argument("--no-seed",  action="store_true",
                        help="Skip cache seeding (use if already warm)")
    args = parser.parse_args()

    cfg = dict(SCENARIOS[args.scenario])
    if args.duration: cfg["duration"] = args.duration
    if args.users:    cfg["users"]    = args.users

    console.print(f"[bold]PrismLib Benchmark[/bold] — scenario=[cyan]{args.scenario}[/cyan] "
                  f"users={cfg['users']} duration={cfg['duration']}s seed={cfg['seed']:,}")

    if not wait_healthy(args.host):
        console.print(f"[red]✗ App not healthy at {args.host} — aborting[/red]")
        sys.exit(1)

    reset_cache(args.host)
    metrics_before = get_metrics(args.host)

    if not args.no_seed:
        seed_cache(args.host, cfg["seed"])

    ts      = time.strftime("%Y%m%d_%H%M%S")
    prefix  = f"prism_{args.scenario}_{ts}"

    csv_path = run_locust(
        host=args.host,
        users=cfg["users"],
        spawn_rate=cfg.get("spawn_rate", 10),
        duration=cfg["duration"],
        csv_prefix=prefix,
        no_azure=args.no_azure,
    )

    metrics_after = get_metrics(args.host)
    print_report(metrics_before, metrics_after, csv_path)

    # Save JSON report
    report_path = Path("benchmark/results") / f"{prefix}_report.json"
    report = {
        "scenario":       args.scenario,
        "host":           args.host,
        "config":         cfg,
        "metrics_before": metrics_before,
        "metrics_after":  metrics_after,
    }
    report_path.write_text(json.dumps(report, indent=2))
    console.print(f"[dim]Full report: {report_path}[/dim]")


if __name__ == "__main__":
    main()
