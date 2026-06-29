"""
prism.cluster.health — HealthMonitor for cluster nodes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _collect_psutil() -> dict[str, float]:
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu_pct": float(cpu),
            "ram_used_pct": float(mem.percent),
            "disk_used_pct": float(disk.percent),
        }
    except ImportError:
        return {"cpu_pct": 0.0, "ram_used_pct": 0.0, "disk_used_pct": 0.0}


class HealthMonitor:
    """
    Periodically collects host metrics and optional app status, then
    forwards to AlertManager.evaluate_health().
    """

    def __init__(
        self,
        node_id: str,
        *,
        interval_seconds: float = 5.0,
        alerter: Optional[Any] = None,
        status_provider: Optional[Callable[[], dict[str, Any]]] = None,
    ) -> None:
        self.node_id = node_id
        self.interval_seconds = interval_seconds
        self._alerter = alerter
        self._status_provider = status_provider
        self._task: Optional[asyncio.Task] = None
        self._closed = False
        self.last_snapshot: dict[str, Any] = {}

    async def start(self) -> None:
        self._closed = False
        self._task = asyncio.create_task(self._loop(), name=f"health-{self.node_id}")

    async def stop(self) -> None:
        self._closed = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while not self._closed:
            snap = await self.collect()
            self.last_snapshot = snap
            if self._alerter is not None:
                try:
                    await self._alerter.evaluate_health(snap)
                except Exception as exc:
                    logger.warning("HealthMonitor: alert evaluation failed: %s", exc)
            await asyncio.sleep(self.interval_seconds)

    async def collect(self) -> dict[str, Any]:
        base = _collect_psutil()
        base["node_id"] = self.node_id
        base["ts"] = time.time()
        if self._status_provider:
            try:
                extra = self._status_provider()
                if extra:
                    base.update(extra)
            except Exception as exc:
                logger.debug("HealthMonitor: status_provider error: %s", exc)
        return base
