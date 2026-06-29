"""Cluster transport and health monitor tests."""

from __future__ import annotations

import pytest

from prism.cluster.health import HealthMonitor
from prism.cluster.transport import ClusterTransport, TransportMode


class TestClusterTransport:
    def test_status_snapshot(self) -> None:
        transport = ClusterTransport(
            "node-a",
            peers={"b": "http://node-b:8080"},
            mode=TransportMode.DIRECT,
        )
        st = transport.status()
        assert st["node_id"] == "node-a"
        assert st["mode"] == "direct"
        assert "http://node-b:8080" in st["peers"]

    @pytest.mark.asyncio
    async def test_handle_incoming_increments_counter(self) -> None:
        received: list[dict] = []

        async def on_frame(frame: dict) -> None:
            received.append(frame)

        transport = ClusterTransport("n1", peers={}, on_frame=on_frame)
        await transport.handle_incoming({"type": "TOKEN_SYNC", "payload": {}})
        assert transport.frames_received == 1
        assert received[0]["type"] == "TOKEN_SYNC"

    @pytest.mark.asyncio
    async def test_publish_broker_mode_noop(self) -> None:
        transport = ClusterTransport(
            "n1",
            peers={"x": "http://peer:8080"},
            mode=TransportMode.BROKER,
        )
        await transport.publish({"type": "ping"})
        assert transport.frames_sent == 0


@pytest.mark.asyncio
async def test_health_monitor_start_stop() -> None:
    mon = HealthMonitor("node-test", interval_seconds=3600.0)
    snap = await mon.collect()
    assert snap["node_id"] == "node-test"
    await mon.start()
    await mon.stop()
