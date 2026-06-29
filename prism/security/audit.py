"""
prism.security.audit — Structured security audit log.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger("prism.audit")


@dataclass
class AuditEvent:
    """One security-relevant event."""

    event_type: str
    actor: str = ""
    resource: str = ""
    outcome: str = "success"  # success | denied | error
    detail: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class AuditLogger:
    """
    Thread-safe audit logger.

    Writes JSON lines to the ``prism.audit`` logger at INFO level.
    Wire to SIEM via standard log shipping (Fluent Bit, CloudWatch, etc.).
    """

    def __init__(self, service: str = "prismlib") -> None:
        self._service = service
        self._lock = threading.RLock()
        self._events: list[AuditEvent] = []
        self._max_buffer = 10_000

    def log(
        self,
        event_type: str,
        *,
        actor: str = "",
        resource: str = "",
        outcome: str = "success",
        **detail: Any,
    ) -> AuditEvent:
        ev = AuditEvent(
            event_type=event_type,
            actor=actor,
            resource=resource,
            outcome=outcome,
            detail={"service": self._service, **detail},
        )
        with self._lock:
            if len(self._events) >= self._max_buffer:
                self._events = self._events[-(self._max_buffer // 2):]
            self._events.append(ev)
        logger.info(ev.to_json())
        return ev

    def auth_success(self, actor: str, resource: str = "/chorus") -> AuditEvent:
        return self.log("auth.success", actor=actor, resource=resource)

    def auth_denied(self, actor: str, resource: str = "/chorus", reason: str = "") -> AuditEvent:
        return self.log("auth.denied", actor=actor, resource=resource, outcome="denied", reason=reason)

    def rate_limited(self, actor: str, resource: str = "/chorus") -> AuditEvent:
        return self.log("rate_limit.exceeded", actor=actor, resource=resource, outcome="denied")

    def recent(self, limit: int = 100) -> list[AuditEvent]:
        with self._lock:
            return list(self._events[-limit:])
