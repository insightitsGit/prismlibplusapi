"""
prism.cluster.transport — CHORUS frame transport between cluster nodes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class TransportMode(str, Enum):
    DIRECT = "direct"
    BROKER = "broker"


FrameHandler = Callable[[dict[str, Any]], Awaitable[None]]


class ClusterTransport:
    """
    Delivers CHORUS-style JSON frames between cluster peers.

    DIRECT mode uses HTTP POST to peer /chorus/ingest endpoints (as in the
    cluster benchmark).  BROKER mode is reserved for Kafka/NATS adapters.
    """

    def __init__(
        self,
        node_id: str,
        peers: dict[str, str],
        *,
        mode: TransportMode = TransportMode.DIRECT,
        on_frame: Optional[FrameHandler] = None,
    ) -> None:
        self.node_id = node_id
        self.peers = peers
        self.mode = mode
        self._on_frame = on_frame
        self._closed = False
        self.frames_sent = 0
        self.frames_received = 0

    async def publish(self, frame: dict[str, Any]) -> None:
        if self.mode != TransportMode.DIRECT:
            logger.warning("ClusterTransport: broker mode not implemented in OSS")
            return
        try:
            import httpx
        except ImportError:
            logger.warning("ClusterTransport: httpx required for DIRECT mode")
            return

        async with httpx.AsyncClient(timeout=10.0) as client:
            for peer_url in self.peers.values():
                try:
                    await client.post(f"{peer_url.rstrip('/')}/chorus/ingest", json=frame)
                    self.frames_sent += 1
                except Exception as exc:
                    logger.debug("ClusterTransport: publish to %s failed: %s", peer_url, exc)

    async def handle_incoming(self, frame: dict[str, Any]) -> None:
        self.frames_received += 1
        if self._on_frame:
            await self._on_frame(frame)

    async def close(self) -> None:
        self._closed = True

    def status(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "mode": self.mode.value,
            "peers": list(self.peers.values()),
            "frames_sent": self.frames_sent,
            "frames_received": self.frames_received,
        }
