"""
PrismNode — Blue/Green/Orange cluster node.

Color chain:
  GREEN  [ACTIVE]   — serving live traffic, publishing WAL fan-out
  BLUE   [WARM]     — fully synced, instant failover if Green dies
  ORANGE [SYNCING]  — catching up to Blue, next in reserve chain

Failover sequence:
  Green dies → Blue flips ACTIVE → Orange flips WARM → new Orange spins up

The chain always self-heals: there is always exactly one ACTIVE,
one WARM, and one SYNCING node (plus new Orange coming up).

Works in two transport modes:
  - DIRECT:  pure CHORUS gRPC streams, no broker required
  - BROKER:  Kafka / NATS / RabbitMQ handles delivery + sync
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class NodeRole(str, Enum):
    GREEN  = "green"   # active master
    BLUE   = "blue"    # warm standby
    ORANGE = "orange"  # syncing reserve


class NodeState(str, Enum):
    ACTIVE   = "active"    # serving — only GREEN should be ACTIVE
    WARM     = "warm"      # synced and ready — BLUE
    SYNCING  = "syncing"   # catching up — ORANGE
    STARTING = "starting"  # initial boot
    FAILED   = "failed"    # self-detected failure


class PrismNode:
    """
    A single node in the Blue/Green/Orange cluster.

    Every physical container runs one PrismNode. At boot the node
    receives its initial role via config or cluster assignment. From
    that point it self-manages transitions based on heartbeats from
    the node above it in the chain.

    Chain hierarchy:
      GREEN (active) → BLUE watches GREEN → ORANGE watches BLUE

    Each node only watches the one directly above it, keeping the
    dependency graph simple and the failure detection fast.
    """

    def __init__(
        self,
        node_id:          str,
        role:             NodeRole,
        peers:            dict[NodeRole, str],   # role → host:port
        heartbeat_interval: float = 0.5,         # seconds between heartbeats
        failure_timeout:    float = 1.5,         # seconds before declaring peer dead
        transport_mode:     str   = "direct",    # "direct" | "broker"
        broker_url:         str   = "",
        on_role_change:     Optional[Callable[[NodeRole, NodeRole], None]] = None,
    ) -> None:
        self.node_id            = node_id
        self.role               = role
        self.state              = NodeState.STARTING
        self.peers              = peers          # other nodes' addresses
        self.heartbeat_interval = heartbeat_interval
        self.failure_timeout    = failure_timeout
        self.transport_mode     = transport_mode
        self.broker_url         = broker_url
        self._on_role_change    = on_role_change

        # Runtime
        self._tasks:           list[asyncio.Task] = []
        self._closed           = False
        self._last_green_beat  = 0.0   # BLUE uses this
        self._last_blue_beat   = 0.0   # ORANGE uses this
        self._frames_received  = 0
        self._frames_published = 0
        self.started_at        = time.monotonic()

        # Metrics
        self.failovers_triggered = 0
        self.promotions_received = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        logger.info("[%s] Starting as %s", self.node_id, self.role.value.upper())
        self.state = self._initial_state()

        self._tasks = [
            asyncio.create_task(self._heartbeat_sender(),  name=f"{self.node_id}-hb-send"),
            asyncio.create_task(self._heartbeat_watcher(), name=f"{self.node_id}-hb-watch"),
            asyncio.create_task(self._sync_loop(),         name=f"{self.node_id}-sync"),
        ]
        logger.info("[%s] %s node is %s", self.node_id, self.role.value.upper(), self.state.value.upper())

    async def close(self) -> None:
        self._closed = True
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("[%s] Node shut down", self.node_id)

    def _initial_state(self) -> NodeState:
        return {
            NodeRole.GREEN:  NodeState.ACTIVE,
            NodeRole.BLUE:   NodeState.WARM,
            NodeRole.ORANGE: NodeState.SYNCING,
        }[self.role]

    # ------------------------------------------------------------------
    # Heartbeat — sender
    # Every node broadcasts its heartbeat to all peers so they know
    # it's alive.
    # ------------------------------------------------------------------

    async def _heartbeat_sender(self) -> None:
        while not self._closed:
            try:
                await self._broadcast_heartbeat()
            except Exception as exc:
                logger.debug("[%s] Heartbeat send error: %s", self.node_id, exc)
            await asyncio.sleep(self.heartbeat_interval)

    async def _broadcast_heartbeat(self) -> None:
        beat = {
            "node_id":    self.node_id,
            "role":       self.role.value,
            "state":      self.state.value,
            "ts":         time.monotonic(),
            "frames_rx":  self._frames_received,
            "frames_tx":  self._frames_published,
        }
        # In direct mode: send via CHORUS gRPC to peers
        # In broker mode: publish to cluster.heartbeat topic
        # Stub — wired up by CHORUSTransport / BrokerTransport
        logger.debug("[%s] ♥ beat sent (%s/%s)", self.node_id, self.role.value, self.state.value)
        _ = beat  # transport layer picks this up

    # ------------------------------------------------------------------
    # Heartbeat — watcher
    # Each node watches exactly one upstream node:
    #   BLUE   watches GREEN
    #   ORANGE watches BLUE
    #   GREEN  watches nothing (it IS the active master)
    # ------------------------------------------------------------------

    async def _heartbeat_watcher(self) -> None:
        if self.role == NodeRole.GREEN:
            return  # active master has nothing to watch

        watch_role = NodeRole.GREEN if self.role == NodeRole.BLUE else NodeRole.BLUE

        while not self._closed:
            await asyncio.sleep(self.heartbeat_interval)
            elapsed = self._elapsed_since_last_beat(watch_role)

            if elapsed > self.failure_timeout:
                logger.warning(
                    "[%s] %s node silent for %.2fs — triggering failover",
                    self.node_id, watch_role.value.upper(), elapsed,
                )
                await self._trigger_failover(watch_role)

    def _elapsed_since_last_beat(self, role: NodeRole) -> float:
        ref = self._last_green_beat if role == NodeRole.GREEN else self._last_blue_beat
        if ref == 0.0:
            return 0.0  # haven't started watching yet
        return time.monotonic() - ref

    def record_heartbeat(self, from_role: NodeRole) -> None:
        """Called by transport layer when a heartbeat frame arrives."""
        if from_role == NodeRole.GREEN:
            self._last_green_beat = time.monotonic()
        elif from_role == NodeRole.BLUE:
            self._last_blue_beat = time.monotonic()

    # ------------------------------------------------------------------
    # Failover
    # ------------------------------------------------------------------

    async def _trigger_failover(self, dead_role: NodeRole) -> None:
        self.failovers_triggered += 1
        old_role = self.role

        if dead_role == NodeRole.GREEN and self.role == NodeRole.BLUE:
            await self._promote_to(NodeRole.GREEN, NodeState.ACTIVE)
            await self._notify_cluster(f"blue_promoted_to_green")
            await self._request_new_orange()

        elif dead_role == NodeRole.BLUE and self.role == NodeRole.ORANGE:
            await self._promote_to(NodeRole.BLUE, NodeState.WARM)
            await self._notify_cluster("orange_promoted_to_blue")
            await self._request_new_orange()

    async def _promote_to(self, new_role: NodeRole, new_state: NodeState) -> None:
        old_role  = self.role
        self.role  = new_role
        self.state = new_state
        self.promotions_received += 1
        logger.info(
            "[%s] ⚡ PROMOTED: %s → %s (%s)",
            self.node_id, old_role.value.upper(), new_role.value.upper(), new_state.value.upper(),
        )
        if self._on_role_change:
            self._on_role_change(old_role, new_role)

    async def _notify_cluster(self, event: str) -> None:
        logger.info("[%s] 📢 Cluster event: %s", self.node_id, event)
        # Transport layer broadcasts this to all followers so they
        # update their routing tables immediately.

    async def _request_new_orange(self) -> None:
        logger.info("[%s] 🟠 Requesting new ORANGE node from orchestrator", self.node_id)
        # In k8s: patch the StatefulSet replicas or trigger a Job
        # In Docker Compose: signal the compose manager
        # Stub — wired up by the orchestrator integration layer

    # ------------------------------------------------------------------
    # Sync loop
    # GREEN  → publishes WAL fan-out to all followers
    # BLUE   → receives from GREEN, mirrors state, forwards to ORANGE
    # ORANGE → receives from BLUE, builds local index
    # ------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        if self.role == NodeRole.GREEN:
            await self._run_as_active()
        elif self.role == NodeRole.BLUE:
            await self._run_as_warm()
        elif self.role == NodeRole.ORANGE:
            await self._run_as_syncing()

    async def _run_as_active(self) -> None:
        logger.info("[%s] 🟢 GREEN: publishing WAL fan-out", self.node_id)
        while not self._closed and self.role == NodeRole.GREEN:
            # Receive WAL events from prism-wrapper via CHORUS / broker
            # Fan out to BLUE and all follower PrismDriver nodes
            await asyncio.sleep(0.01)

    async def _run_as_warm(self) -> None:
        logger.info("[%s] 🔵 BLUE: mirroring GREEN state", self.node_id)
        while not self._closed and self.role == NodeRole.BLUE:
            # Receive mirror stream from GREEN
            # Apply same frames to local index (stay identical to GREEN)
            # Forward to ORANGE so it can sync too
            await asyncio.sleep(0.01)

    async def _run_as_syncing(self) -> None:
        logger.info("[%s] 🟠 ORANGE: catching up to BLUE", self.node_id)
        while not self._closed and self.role == NodeRole.ORANGE:
            # Receive sync stream from BLUE
            # Build local index — state = SYNCING until caught up
            # Once caught up: flip state to WARM
            await asyncio.sleep(0.01)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> dict:
        return {
            "node_id":            self.node_id,
            "role":               self.role.value,
            "state":              self.state.value,
            "transport":          self.transport_mode,
            "frames_received":    self._frames_received,
            "frames_published":   self._frames_published,
            "failovers_triggered": self.failovers_triggered,
            "promotions_received": self.promotions_received,
            "uptime_s":           round(time.monotonic() - self.started_at, 1),
        }
