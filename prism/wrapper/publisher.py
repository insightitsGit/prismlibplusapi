"""
prism.wrapper.publisher — CHORUS Fabric publisher.

Receives vectorized WALEvents from the interceptor and streams them to
registered DLL Driver endpoints over the CHORUS Fabric gRPC transport.

Design:
  - One asyncio.Queue fed by the interceptor
  - N concurrent publisher coroutines (one per connected DLL Driver)
  - Exponential backoff on transient gRPC errors
  - TensorCipher key rotation is handled transparently
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from prism.lib.fabric import CHORUSFabric, FabricConfig, VectorFrame
from prism.wrapper.interceptor import WALEvent, WALEventType, RowVectorizer
from prism.wrapper.row_events import RowEventHub, RowEventRecord

logger = logging.getLogger(__name__)

_BACKOFF_BASE = 1.0
_BACKOFF_MAX  = 60.0


@dataclass
class DriverEndpoint:
    """A registered DLL Driver that this wrapper streams events to."""
    endpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    host: str = "localhost"
    port: int = 50052
    tenant_id: str = ""
    tls_cert_path: Optional[str] = None
    connected_at: float = field(default_factory=time.monotonic)


class CHORUSPublisher:
    """
    Consumes WALEvents from an asyncio.Queue, vectorizes them (if not already),
    and fans them out to all registered DLL Driver endpoints via CHORUS Fabric.

    Usage:
        publisher = CHORUSPublisher(tenant_id="acme", target_dim=64)
        publisher.register_driver(DriverEndpoint(host="app-node-1", port=50052))
        await publisher.run(event_queue)
    """

    def __init__(
        self,
        tenant_id: str,
        target_dim: int = 64,
        key_ttl_seconds: float = 300.0,
        publish_batch_size: int = 64,
        event_hub: Optional[RowEventHub] = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._vectorizer = RowVectorizer(tenant_id=tenant_id, target_dim=target_dim)
        self._key_ttl = key_ttl_seconds
        self._batch_size = publish_batch_size
        self._hub = event_hub or RowEventHub()

        self._drivers: dict[str, DriverEndpoint] = {}
        self._fabrics: dict[str, CHORUSFabric] = {}

    @property
    def event_hub(self) -> RowEventHub:
        return self._hub

    # ------------------------------------------------------------------
    # Driver registry
    # ------------------------------------------------------------------

    def register_driver(self, endpoint: DriverEndpoint) -> str:
        """Add a DLL Driver endpoint. Returns endpoint_id."""
        self._drivers[endpoint.endpoint_id] = endpoint
        logger.info(
            "CHORUSPublisher: registered driver %s at %s:%d",
            endpoint.endpoint_id,
            endpoint.host,
            endpoint.port,
        )
        return endpoint.endpoint_id

    def deregister_driver(self, endpoint_id: str) -> None:
        self._drivers.pop(endpoint_id, None)
        fabric = self._fabrics.pop(endpoint_id, None)
        if fabric is not None:
            asyncio.create_task(fabric.close())
        logger.info("CHORUSPublisher: deregistered driver %s", endpoint_id)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, event_queue: asyncio.Queue[WALEvent]) -> None:
        """
        Consume events from event_queue indefinitely.

        This coroutine runs until cancelled (i.e. until the daemon shuts down).
        Each WALEvent is vectorized and published to all registered drivers.
        """
        logger.info(
            "CHORUSPublisher: running (tenant=%s, batch=%d)",
            self._tenant_id,
            self._batch_size,
        )

        # Connect all registered drivers
        await self._connect_all()

        batch: list[tuple[WALEvent, np.ndarray, str]] = []

        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.05)
                self._publish_row_event(event)
                if event.event_type != WALEventType.DELETE:
                    text_repr, vector = self._vectorizer.vectorize(event)
                    batch.append((event, vector, text_repr))
                event_queue.task_done()
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                if batch:
                    await self._flush_batch(batch)
                raise

            if len(batch) >= self._batch_size:
                await self._flush_batch(batch)
                batch = []

    async def _flush_batch(
        self,
        batch: list[tuple[WALEvent, np.ndarray, str]],
    ) -> None:
        """Publish a collected batch to all registered drivers."""
        if not batch:
            return

        vectors = np.stack([v for _, v, _ in batch]).astype(np.float32)

        tasks = [
            self._publish_to_driver(ep_id, vectors)
            for ep_id in list(self._drivers.keys())
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for ep_id, result in zip(self._drivers.keys(), results):
            if isinstance(result, Exception):
                logger.warning(
                    "CHORUSPublisher: publish to driver %s failed: %s",
                    ep_id,
                    result,
                )

        logger.debug(
            "CHORUSPublisher: flushed batch of %d events to %d drivers",
            len(batch),
            len(self._drivers),
        )

    async def _publish_to_driver(self, endpoint_id: str, vectors: np.ndarray) -> None:
        """Send a vector batch to one driver with exponential backoff retry."""
        fabric = await self._get_or_connect(endpoint_id)
        backoff = _BACKOFF_BASE

        for attempt in range(5):
            try:
                await fabric.send(vectors)
                return
            except Exception as exc:
                logger.warning(
                    "CHORUSPublisher: send attempt %d to %s failed: %s — retry in %.1fs",
                    attempt + 1,
                    endpoint_id,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

                # Reconnect on persistent failure
                try:
                    await fabric.close()
                except Exception:
                    pass
                self._fabrics.pop(endpoint_id, None)
                fabric = await self._get_or_connect(endpoint_id)

        raise RuntimeError(f"All publish attempts to driver {endpoint_id} exhausted.")

    async def _get_or_connect(self, endpoint_id: str) -> CHORUSFabric:
        if endpoint_id in self._fabrics:
            return self._fabrics[endpoint_id]
        return await self._connect_driver(endpoint_id)

    async def _connect_driver(self, endpoint_id: str) -> CHORUSFabric:
        ep = self._drivers[endpoint_id]
        cfg = FabricConfig(
            host=ep.host,
            port=ep.port,
            key_ttl_seconds=self._key_ttl,
            tls_cert_path=ep.tls_cert_path,
        )
        fabric = CHORUSFabric(cfg)
        await fabric.connect()
        self._fabrics[endpoint_id] = fabric
        logger.info(
            "CHORUSPublisher: connected fabric to driver %s (%s:%d)",
            endpoint_id,
            ep.host,
            ep.port,
        )
        return fabric

    async def _connect_all(self) -> None:
        tasks = [self._connect_driver(ep_id) for ep_id in self._drivers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for ep_id, result in zip(self._drivers, results):
            if isinstance(result, Exception):
                logger.warning(
                    "CHORUSPublisher: initial connect to %s failed: %s", ep_id, result
                )

    async def close(self) -> None:
        tasks = [fabric.close() for fabric in self._fabrics.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._fabrics.clear()
        logger.info("CHORUSPublisher: all fabric connections closed.")

    def _publish_row_event(self, event: WALEvent) -> None:
        """Publish a typed row event to the subscribe hub for drivers."""
        op = event.event_type.value
        vector: Optional[list[float]] = None
        text_repr = ""
        if event.event_type != WALEventType.DELETE:
            text_repr, vec = self._vectorizer.vectorize(event)
            vector = vec.tolist()
        self._hub.publish(RowEventRecord(
            event_id=event.event_id,
            row_id=event.row_id,
            op=op,
            text_repr=text_repr,
            vector=vector,
            table_name=event.table_name,
        ))
