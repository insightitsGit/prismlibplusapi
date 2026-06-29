"""
Enterprise readiness tests — LocalIndex WAL ops, auth, cluster, alerts.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism.api.auth import AuthConfig, generate_api_key
from prism.cluster.alerts import AlertRule, AlertLevel, _evaluate_condition
from prism.cluster.health import HealthMonitor
from prism.cluster.transport import ClusterTransport, TransportMode
from prism.ffi.bindings import LocalIndex
from prism.wrapper.interceptor import WALEvent, WALEventType
from prism.wrapper.publisher import CHORUSPublisher
from prism.wrapper.row_events import RowEventHub


class TestLocalIndexWALOps:
    def test_upsert_and_query(self) -> None:
        idx = LocalIndex("t1", dim=4)
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0]
        idx.apply_event(event_id="e1", row_id="r1", op="INSERT", text_repr="a", vector=v1)
        idx.apply_event(event_id="e2", row_id="r2", op="INSERT", text_repr="b", vector=v2)

        results, _ = idx.query(np.array(v1, dtype=np.float32), top_k=2, threshold=0.5)
        assert len(results) >= 1
        assert results[0].row_id == "r1"
        assert idx.size == 2

    def test_update_replaces_row(self) -> None:
        idx = LocalIndex("t1", dim=4)
        v_old = [1.0, 0.0, 0.0, 0.0]
        v_new = [0.0, 0.0, 1.0, 0.0]
        idx.apply_event(event_id="e1", row_id="doc-1", op="INSERT", text_repr="old", vector=v_old)
        idx.apply_event(event_id="e2", row_id="doc-1", op="UPDATE", text_repr="new", vector=v_new)

        assert idx.size == 1
        results, _ = idx.query(np.array(v_new, dtype=np.float32), top_k=1, threshold=0.5)
        assert results[0].text_repr == "new"

    def test_delete_removes_row(self) -> None:
        idx = LocalIndex("t1", dim=4)
        v = [1.0, 0.0, 0.0, 0.0]
        idx.apply_event(event_id="e1", row_id="gone", op="INSERT", text_repr="x", vector=v)
        assert idx.size == 1
        assert idx.is_warm
        idx.apply_event(event_id="e2", row_id="gone", op="DELETE", text_repr="")
        assert idx.size == 0
        assert not idx.is_warm
        results, _ = idx.query(np.array(v, dtype=np.float32), top_k=5, threshold=0.1)
        assert results == []


class TestRowEventHub:
    def test_publish_and_snapshot(self) -> None:
        from prism.wrapper.row_events import RowEventRecord

        hub = RowEventHub()
        hub.publish(RowEventRecord("e1", "r1", "INSERT", vector=[1.0, 0.0]))
        assert len(hub.snapshot()) == 1


class TestPublisherDelete:
    def test_delete_publishes_without_vector(self) -> None:
        hub = RowEventHub()
        pub = CHORUSPublisher(tenant_id="t", event_hub=hub)
        ev = WALEvent(
            event_id="e1",
            table_name="docs",
            event_type=WALEventType.DELETE,
            before={"id": "doc-99", "title": "gone"},
            after=None,
        )
        pub._publish_row_event(ev)
        snap = hub.snapshot()
        assert len(snap) == 1
        assert snap[0].op == "DELETE"
        assert snap[0].row_id == "doc-99"
        assert snap[0].vector is None


class TestAuth:
    def test_api_key_validation(self) -> None:
        key = generate_api_key()
        cfg = AuthConfig(api_keys=(key,), require_auth=True)
        assert cfg.validate_headers({"X-API-Key": key})
        assert not cfg.validate_headers({"X-API-Key": "wrong"})

    def test_bearer_validation(self) -> None:
        token = generate_api_key()
        cfg = AuthConfig(bearer_tokens=(token,), require_auth=True)
        assert cfg.validate_headers({"Authorization": f"Bearer {token}"})
        assert not cfg.validate_headers({"Authorization": "Bearer bad"})


class TestSafeAlertRules:
    def test_condition_eval(self) -> None:
        assert _evaluate_condition("cpu_pct > 90", {"cpu_pct": 95})
        assert not _evaluate_condition("cpu_pct > 90", {"cpu_pct": 50})

    def test_alert_rule_fires(self) -> None:
        rule = AlertRule("cpu", "cpu_pct > 90", AlertLevel.WARNING, cooldown_s=0)
        assert rule.should_fire({"cpu_pct": 95})
        assert not rule.should_fire({"cpu_pct": 10})


class TestClusterModulesImport:
    def test_cluster_package_imports(self) -> None:
        from prism.cluster import HealthMonitor, ClusterTransport, ClusterCache

        assert HealthMonitor is not None
        assert ClusterTransport is not None
        assert ClusterCache is not None


class TestObservability:
    def test_prometheus_export(self) -> None:
        from prism.observability import get_registry, record_cache_hit
        from prism.observability.prometheus import prometheus_handler

        record_cache_hit(latency_ms=1.5)
        body, ctype = prometheus_handler()
        assert b"prism_cache_hits_total" in body
        assert "text/plain" in ctype


@pytest.mark.asyncio
async def test_health_monitor_collect() -> None:
    mon = HealthMonitor("node-1", interval_seconds=60.0)
    snap = await mon.collect()
    assert snap["node_id"] == "node-1"
    assert "cpu_pct" in snap
