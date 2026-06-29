"""
Cluster benchmark node — FastAPI app that runs a PrismNode.

Each container runs this app. Role (green/blue/orange) is set via env var.
All 4 nodes talk to each other through CHORUS HTTP tunnels.

Environment variables:
  NODE_ID        — unique name e.g. "node-green"
  NODE_ROLE      — green | blue | orange
  PEERS          — JSON: {"green":"http://...", "blue":"http://...", "orange":"http://..."}
  ADMIN_EMAIL    — comma-separated alert recipients
  SMTP_HOST      — mail server (optional)
  SMTP_USER      — mail username (optional)
  SMTP_PASS      — mail password (optional)
  TOKEN_BUDGET   — daily token budget for alerts (default 100000)
  NETWORK_LABEL  — "same-pod" | "cross-network" (for benchmark labelling)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
import numpy as np
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging — structured JSON so Azure Log Analytics can parse it
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","node":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger(os.getenv("NODE_ID", "prism-node"))


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

NODE_ID       = os.getenv("NODE_ID", f"node-{uuid.uuid4().hex[:6]}")
NODE_ROLE     = os.getenv("NODE_ROLE", "green")
NETWORK_LABEL = os.getenv("NETWORK_LABEL", "unknown")
TOKEN_BUDGET  = int(os.getenv("TOKEN_BUDGET", "100000"))
ADMIN_EMAIL   = [e.strip() for e in os.getenv("ADMIN_EMAIL", "").split(",") if e.strip()]
SMTP_HOST     = os.getenv("SMTP_HOST", "")
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASS     = os.getenv("SMTP_PASS", "")

# Peer URLs — JSON dict: {"green": "https://...", "blue": "https://...", "orange": "https://..."}
_peers_raw = os.getenv("PEERS", "{}")
try:
    PEERS: dict[str, str] = json.loads(_peers_raw)
except Exception:
    PEERS = {}


# ---------------------------------------------------------------------------
# In-memory cluster state
# ---------------------------------------------------------------------------

class ClusterState:
    """Shared state for this node — mutated by API calls and background tasks."""

    def __init__(self) -> None:
        self.heartbeat_paused = False   # set True by simulate_failover
        self.role        = NODE_ROLE
        self.state       = "active" if NODE_ROLE == "green" else \
                           "warm"   if NODE_ROLE == "blue"  else "syncing"
        self.started_at  = time.time()

        # Cluster cache — shared answers from other nodes
        self.cluster_cache:  dict[str, dict] = {}   # query_hash → {answer, tokens, source, ts}

        # Token tracking
        self.tokens_billed       = 0
        self.tokens_cached       = 0      # served from local cache
        self.tokens_cluster_hit  = 0      # served from another node's cache
        self.tokens_compressed   = 0      # saved by context compression
        self.tokens_deduped      = 0      # saved by in-flight dedup
        self.llm_calls_made      = 0
        self.llm_calls_avoided   = 0

        # CHORUS frame tracking
        self.frames_sent     = 0
        self.frames_received = 0
        self.frame_latencies: list[float] = []  # ms

        # Health history (last 60 snapshots)
        self.health_history: list[dict] = []

        # Alert log
        self.alerts_fired: list[dict] = []

        # Heartbeat tracking — when did we last hear from each peer
        self.peer_last_seen: dict[str, float] = {}

        # In-flight dedup
        self._in_flight: dict[str, asyncio.Future] = {}

        # Failover tracking
        self.failovers_triggered = 0
        self.last_failover_at:  Optional[float] = None
        self.failover_duration_ms: Optional[float] = None

    @property
    def uptime_s(self) -> float:
        return time.time() - self.started_at

    @property
    def token_savings_pct(self) -> float:
        total = self.tokens_billed + self.tokens_cached + \
                self.tokens_cluster_hit + self.tokens_compressed + self.tokens_deduped
        saved = self.tokens_cached + self.tokens_cluster_hit + \
                self.tokens_compressed + self.tokens_deduped
        return (saved / total * 100) if total > 0 else 0.0

    @property
    def estimated_cost_usd(self) -> float:
        return self.tokens_billed / 1_000_000 * 5.0   # $5/1M tokens baseline

    @property
    def estimated_savings_usd(self) -> float:
        saved = self.tokens_cached + self.tokens_cluster_hit + \
                self.tokens_compressed + self.tokens_deduped
        return saved / 1_000_000 * 5.0

    def capture_health(self) -> dict:
        proc  = psutil.Process()
        mem   = psutil.virtual_memory()
        disk  = psutil.disk_usage("/")
        snap  = {
            "node_id":        NODE_ID,
            "role":           self.role,
            "state":          self.state,
            "network":        NETWORK_LABEL,
            "ts":             time.time(),
            "uptime_s":       round(self.uptime_s, 1),
            "cpu_pct":        psutil.cpu_percent(interval=0.1),
            "ram_used_mb":    round(proc.memory_info().rss / 1024 / 1024, 1),
            "ram_total_mb":   round(mem.total / 1024 / 1024, 1),
            "ram_used_pct":   round(mem.percent, 1),
            "disk_used_gb":   round(disk.used  / 1024**3, 2),
            "disk_total_gb":  round(disk.total / 1024**3, 2),
            "disk_used_pct":  round(disk.percent, 1),
            "frames_sent":    self.frames_sent,
            "frames_received":self.frames_received,
            "avg_frame_ms":   round(sum(self.frame_latencies[-100:]) /
                                    max(len(self.frame_latencies[-100:]), 1), 2),
            "cluster_cache_size": len(self.cluster_cache),
            "tokens_billed":  self.tokens_billed,
            "tokens_saved_total": self.tokens_cached + self.tokens_cluster_hit +
                                  self.tokens_compressed + self.tokens_deduped,
            "token_savings_pct":  round(self.token_savings_pct, 1),
            "llm_calls_made": self.llm_calls_made,
            "llm_calls_avoided": self.llm_calls_avoided,
            "failovers":      self.failovers_triggered,
            "peers_seen":     {k: round(time.time() - v, 1)
                               for k, v in self.peer_last_seen.items()},
        }
        self.health_history.append(snap)
        if len(self.health_history) > 60:
            self.health_history.pop(0)
        return snap


_state = ClusterState()


# ---------------------------------------------------------------------------
# Mock LLM — counts tokens, simulates cost
# ---------------------------------------------------------------------------

SAMPLE_ANSWERS = [
    "PrismLib reduces LLM costs by caching semantically equivalent queries in-process.",
    "CHORUS Fabric streams WAL events as encrypted float32 frames via gRPC.",
    "The ClusterCache shares answers across nodes to eliminate redundant LLM calls.",
    "Blue/Green/Orange topology ensures zero-downtime failover with 50ms switchover.",
    "Context compression keeps only the top-K relevant chunks, reducing tokens by 90%.",
    "JL projection seeded by SHA-256(tenant_id) provides mathematical cross-tenant isolation.",
    "PrismResonance uses wave-interference similarity for sub-millisecond lookups.",
    "The HEALTH frame type carries container vitals across the tunnel to all nodes.",
    "AlertManager fires emails and SIGNAL frames when health thresholds are crossed.",
    "Token deduplication coalesces identical in-flight queries into one LLM call.",
]

def mock_llm(query: str, context_tokens: int = 500) -> tuple[str, int, int]:
    """Returns (answer, prompt_tokens, completion_tokens). Simulates 80ms latency."""
    time.sleep(0.08)
    idx    = int(hashlib.sha256(query.encode()).hexdigest(), 16) % len(SAMPLE_ANSWERS)
    answer = SAMPLE_ANSWERS[idx]
    prompt_tokens     = len(query.split()) + context_tokens
    completion_tokens = len(answer.split()) * 2
    return answer, prompt_tokens, completion_tokens


def make_query_vector(text: str, dim: int = 64) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng  = np.random.default_rng(seed)
    v    = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------------------
# CHORUS tunnel — simplified HTTP implementation for benchmark
# ---------------------------------------------------------------------------

async def broadcast_frame(frame_type: str, payload: dict) -> dict[str, float]:
    """
    Send a CHORUS frame to all peers. Returns {peer_role: latency_ms}.
    In production this is gRPC binary streaming; here we use HTTP JSON
    so the benchmark works without compiled proto stubs.
    """
    latencies: dict[str, float] = {}
    frame = {
        "frame_type": frame_type,
        "source_node": NODE_ID,
        "source_role": _state.role,
        "seq": _state.frames_sent,
        "ts": time.time(),
        "payload": payload,
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        for role, url in PEERS.items():
            if not url:
                continue
            t0 = time.monotonic()
            try:
                r = await client.post(f"{url}/chorus/ingest", json=frame)
                r.raise_for_status()
                latency_ms = (time.monotonic() - t0) * 1000
                latencies[role] = round(latency_ms, 2)
                _state.frame_latencies.append(latency_ms)
            except Exception as exc:
                logger.warning("Frame to %s failed: %s", role, exc)
    _state.frames_sent += 1
    return latencies


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

async def fire_alert(level: str, event_type: str, title: str, message: str, data: dict) -> None:
    record = {
        "node_id": NODE_ID, "level": level, "event_type": event_type,
        "title": title, "message": message, "data": data, "ts": time.time(),
    }
    _state.alerts_fired.append(record)
    logger.warning("ALERT [%s] %s: %s | data=%s", level.upper(), event_type, message, data)

    # Broadcast SIGNAL frame to all peers
    await broadcast_frame("SIGNAL", {
        "signal_type": event_type, "severity": level,
        "description": message, "data": data,
    })

    # Email — if configured
    if ADMIN_EMAIL and SMTP_HOST:
        await asyncio.get_event_loop().run_in_executor(None, _send_alert_email,
                                                       level, title, message, data)


def _send_alert_email(level: str, title: str, message: str, data: dict) -> None:
    import smtplib, ssl
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    try:
        emoji   = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(level, "🔔")
        subject = f"{emoji} [{level.upper()}] PrismLib Cluster — {title}"
        body    = f"{message}\n\nNode: {NODE_ID}\nRole: {NODE_ROLE}\nData: {json.dumps(data, indent=2)}"
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = ", ".join(ADMIN_EMAIL)
        msg.attach(MIMEText(body, "plain"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, 587) as s:
            s.starttls(context=ctx)
            if SMTP_USER and SMTP_PASS:
                s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_USER, ADMIN_EMAIL, msg.as_string())
        logger.info("Alert email sent to %s", ADMIN_EMAIL)
    except Exception as exc:
        logger.error("Alert email failed: %s", exc)


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def heartbeat_loop() -> None:
    """Send HEALTH frame to all peers every 2 seconds."""
    while True:
        try:
            if _state.heartbeat_paused:
                await asyncio.sleep(2)
                continue
            health = _state.capture_health()
            await broadcast_frame("HEALTH", health)

            # Check alert rules
            if health["cpu_pct"] > 85:
                await fire_alert("warning", "cpu_high",
                    f"CPU {health['cpu_pct']:.0f}% on {NODE_ID}",
                    f"Node {NODE_ID} CPU above 85%", health)
            if health["ram_used_pct"] > 85:
                await fire_alert("warning", "ram_high",
                    f"RAM {health['ram_used_pct']:.0f}% on {NODE_ID}",
                    f"Node {NODE_ID} RAM above 85%", health)
        except Exception as exc:
            logger.debug("Heartbeat error: %s", exc)
        await asyncio.sleep(2)


async def peer_watchdog_loop() -> None:
    """
    Watch peer heartbeats. If GREEN goes silent for >3s and we are BLUE,
    trigger failover: BLUE → GREEN, notify cluster.
    """
    await asyncio.sleep(5)   # let peers boot first
    while True:
        await asyncio.sleep(1)
        now = time.time()
        for role, last_seen in list(_state.peer_last_seen.items()):
            silence = now - last_seen
            if silence > 3.0:
                if role == "green" and _state.role == "blue":
                    logger.warning("GREEN silent %.1fs — triggering failover", silence)
                    t_start = time.monotonic()
                    _state.role   = "green"
                    _state.state  = "active"
                    _state.failovers_triggered += 1
                    _state.last_failover_at    = time.time()
                    await broadcast_frame("SIGNAL", {
                        "signal_type": "failover",
                        "severity":    "warning",
                        "description": f"{NODE_ID} promoted GREEN after {silence:.1f}s silence",
                        "new_green":   NODE_ID,
                    })
                    await fire_alert("warning", "failover_triggered",
                        f"{NODE_ID} promoted to GREEN",
                        f"GREEN node silent {silence:.1f}s. {NODE_ID} is now active master.",
                        {"silence_s": round(silence, 2), "new_green": NODE_ID})
                    _state.failover_duration_ms = (time.monotonic() - t_start) * 1000
                    logger.info("Failover complete in %.1fms", _state.failover_duration_ms)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Node %s starting as %s (%s)", NODE_ID, NODE_ROLE.upper(), NETWORK_LABEL)
    bg = [
        asyncio.create_task(heartbeat_loop()),
        asyncio.create_task(peer_watchdog_loop()),
    ]
    yield
    for t in bg:
        t.cancel()
    logger.info("Node %s shutdown", NODE_ID)


app = FastAPI(title=f"PrismNode [{NODE_ROLE.upper()}]", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query:          str
    context_chunks: list[str] = []
    use_cluster_cache: bool   = True
    use_compression:   bool   = True


class ChorusFrame(BaseModel):
    frame_type:  str
    source_node: str
    source_role: str
    seq:         int
    ts:          float
    payload:     dict


# ---------------------------------------------------------------------------
# Main query endpoint — full token-saving pipeline
# ---------------------------------------------------------------------------

@app.post("/query")
async def query(req: QueryRequest):
    t0          = time.monotonic()
    query_hash  = hashlib.sha256(req.query.encode()).hexdigest()
    query_vec   = make_query_vector(req.query)

    # ---------- Step 1: cluster cache check ----------
    if req.use_cluster_cache and query_hash in _state.cluster_cache:
        entry = _state.cluster_cache[query_hash]
        if time.time() - entry["ts"] < 3600:
            _state.tokens_cluster_hit += entry["tokens"]
            _state.llm_calls_avoided  += 1
            return {
                "answer":      entry["answer"],
                "source":      f"cluster_cache (from {entry['source']})",
                "tokens_used": 0,
                "tokens_saved": entry["tokens"],
                "latency_ms":  round((time.monotonic() - t0) * 1000, 2),
                "savings_type": "cluster_cache",
            }

    # ---------- Step 2: context compression ----------
    context_tokens = 500   # baseline
    compressed_tokens = context_tokens
    tokens_compressed = 0

    if req.use_compression and req.context_chunks:
        # Keep top-3 most relevant chunks by cosine sim with query vector
        scores = []
        for chunk in req.context_chunks:
            cv    = make_query_vector(chunk)
            score = float(np.dot(query_vec, cv))
            scores.append((score, chunk))
        scores.sort(reverse=True)
        kept_chunks    = [c for _, c in scores[:3]]
        orig_tokens    = sum(len(c.split()) * 2 for c in req.context_chunks)
        comp_tokens    = sum(len(c.split()) * 2 for c in kept_chunks)
        tokens_compressed      = max(0, orig_tokens - comp_tokens)
        compressed_tokens      = comp_tokens
        _state.tokens_compressed += tokens_compressed

    # ---------- Step 3: in-flight dedup ----------
    if query_hash in _state._in_flight:
        logger.debug("Dedup: coalescing onto in-flight call %s", query_hash[:8])
        answer, pt, ct = await asyncio.shield(_state._in_flight[query_hash])
        _state.tokens_deduped   += pt + ct
        _state.llm_calls_avoided += 1
        return {
            "answer":       answer,
            "source":       "dedup (coalesced)",
            "tokens_used":  0,
            "tokens_saved": pt + ct,
            "latency_ms":   round((time.monotonic() - t0) * 1000, 2),
            "savings_type": "dedup",
        }

    # ---------- Step 4: LLM call ----------
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    _state._in_flight[query_hash] = future
    try:
        answer, pt, ct = await asyncio.get_event_loop().run_in_executor(
            None, mock_llm, req.query, compressed_tokens
        )
        future.set_result((answer, pt, ct))
    except Exception as exc:
        future.set_exception(exc)
        raise
    finally:
        _state._in_flight.pop(query_hash, None)

    _state.tokens_billed  += pt + ct
    _state.llm_calls_made += 1

    # ---------- Step 5: broadcast to cluster ----------
    cache_entry = {
        "query_hash": query_hash,
        "answer":     answer,
        "tokens":     pt + ct,
        "source":     NODE_ID,
        "ts":         time.time(),
    }
    _state.cluster_cache[query_hash] = cache_entry
    latencies = await broadcast_frame("TOKEN_SYNC", cache_entry)

    total_saved = tokens_compressed
    return {
        "answer":           answer,
        "source":           "llm_call",
        "tokens_used":      pt + ct,
        "tokens_compressed": tokens_compressed,
        "tokens_saved":     total_saved,
        "latency_ms":       round((time.monotonic() - t0) * 1000, 2),
        "broadcast_ms":     latencies,
        "savings_type":     "compression" if tokens_compressed > 0 else "none",
    }


# ---------------------------------------------------------------------------
# CHORUS ingest — receives frames from other nodes
# ---------------------------------------------------------------------------

@app.post("/chorus/ingest")
async def chorus_ingest(frame: ChorusFrame):
    t_received = time.time()
    _state.frames_received += 1
    _state.peer_last_seen[frame.source_role] = t_received

    transit_ms = round((t_received - frame.ts) * 1000, 2)
    logger.info("CHORUS %s from %s (%s) transit=%.1fms",
                frame.frame_type, frame.source_node, frame.source_role, transit_ms)

    if frame.frame_type == "TOKEN_SYNC":
        p = frame.payload
        if "query_hash" in p and "answer" in p:
            _state.cluster_cache[p["query_hash"]] = {**p, "ts": time.time()}

    elif frame.frame_type == "HEALTH":
        pass  # logged above, peer_last_seen already updated

    elif frame.frame_type == "SIGNAL":
        sig = frame.payload
        logger.warning("SIGNAL [%s] from %s: %s — %s",
                       sig.get("severity","?").upper(),
                       frame.source_node,
                       sig.get("signal_type","?"),
                       sig.get("description",""))
        # Store incoming alerts so /alerts count grows — benchmark polls this
        _state.alerts_fired.append({
            "node_id": NODE_ID, "level": sig.get("severity", "info"),
            "event_type": sig.get("signal_type", "?"), "source": frame.source_node,
            "title": sig.get("description", ""), "message": sig.get("description", ""),
            "data": sig.get("data", {}), "ts": t_received, "propagated": True,
        })
        if sig.get("signal_type") == "failover":
            logger.info("Cluster topology change: new GREEN = %s", sig.get("new_green"))

    elif frame.frame_type == "CONFIG":
        logger.info("CONFIG update from %s: %s", frame.source_node, frame.payload)

    return {"status": "ok", "transit_ms": transit_ms}


# ---------------------------------------------------------------------------
# Status and health endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "node_id": NODE_ID, "role": _state.role, "state": _state.state}


@app.get("/status")
async def status():
    return _state.capture_health()


@app.get("/metrics/tokens")
async def token_metrics():
    total_saved = (_state.tokens_cached + _state.tokens_cluster_hit +
                   _state.tokens_compressed + _state.tokens_deduped)
    return {
        "node_id":              NODE_ID,
        "role":                 _state.role,
        "network":              NETWORK_LABEL,
        "tokens_billed":        _state.tokens_billed,
        "tokens_cached_local":  _state.tokens_cached,
        "tokens_cached_cluster":_state.tokens_cluster_hit,
        "tokens_compressed":    _state.tokens_compressed,
        "tokens_deduped":       _state.tokens_deduped,
        "total_saved":          total_saved,
        "savings_pct":          round(_state.token_savings_pct, 1),
        "estimated_cost_usd":   round(_state.estimated_cost_usd, 4),
        "estimated_savings_usd":round(_state.estimated_savings_usd, 4),
        "llm_calls_made":       _state.llm_calls_made,
        "llm_calls_avoided":    _state.llm_calls_avoided,
        "cluster_cache_size":   len(_state.cluster_cache),
    }


@app.get("/metrics/chorus")
async def chorus_metrics():
    lats = _state.frame_latencies[-500:]
    return {
        "node_id":       NODE_ID,
        "network":       NETWORK_LABEL,
        "frames_sent":   _state.frames_sent,
        "frames_received": _state.frames_received,
        "avg_latency_ms": round(sum(lats) / max(len(lats), 1), 2),
        "min_latency_ms": round(min(lats), 2) if lats else 0,
        "max_latency_ms": round(max(lats), 2) if lats else 0,
        "p99_latency_ms": round(sorted(lats)[int(len(lats) * 0.99)] if len(lats) > 10 else 0, 2),
    }


@app.get("/metrics/failover")
async def failover_metrics():
    return {
        "node_id":             NODE_ID,
        "current_role":        _state.role,
        "failovers_triggered": _state.failovers_triggered,
        "last_failover_at":    _state.last_failover_at,
        "failover_duration_ms":_state.failover_duration_ms,
    }


@app.get("/alerts")
async def alerts():
    return {"node_id": NODE_ID, "alerts": _state.alerts_fired[-20:]}


@app.get("/health/history")
async def health_history():
    return {"node_id": NODE_ID, "snapshots": _state.health_history[-20:]}


@app.post("/admin/reset")
async def reset():
    _state.cluster_cache.clear()
    _state.tokens_billed = _state.tokens_cached = _state.tokens_cluster_hit = 0
    _state.tokens_compressed = _state.tokens_deduped = 0
    _state.llm_calls_made = _state.llm_calls_avoided = 0
    _state.frames_sent = _state.frames_received = 0
    _state.frame_latencies.clear()
    _state.alerts_fired.clear()
    _state.failovers_triggered = 0
    _state.failover_duration_ms = None
    return {"status": "reset", "node_id": NODE_ID}


@app.post("/admin/fire_test_alert")
async def fire_test_alert():
    """Trigger a real alert that broadcasts SIGNAL frames to all peers."""
    await fire_alert(
        level="warning", event_type="cpu_high",
        title="Benchmark test alert — cpu_high",
        message=f"Benchmark injected: CPU 92% on {NODE_ID}",
        data={"cpu_pct": 92, "node": NODE_ID, "benchmark": True},
    )
    return {"status": "alert_fired", "node_id": NODE_ID}


@app.post("/admin/simulate_failover")
async def simulate_failover():
    """Force this node to stop sending heartbeats for 5s — triggers BLUE failover."""
    async def _pause():
        logger.warning("Simulating GREEN failure — pausing heartbeats for 5s")
        _state.heartbeat_paused = True
        await asyncio.sleep(5)
        _state.heartbeat_paused = False
        logger.info("Heartbeat resumed")
    asyncio.create_task(_pause())
    return {"status": "heartbeat_paused_5s", "node_id": NODE_ID}


if __name__ == "__main__":
    uvicorn.run("node_app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")),
                workers=1, log_level="info")
