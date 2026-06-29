"""
prism.wrapper.row_events — WAL row event hub for driver subscriptions.

The CHORUSPublisher writes fully-typed row events (INSERT/UPDATE/DELETE)
into RowEventHub.  SubscribeHTTPServer streams them to PrismDriver clients.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional


@dataclass(frozen=True)
class RowEventRecord:
    """One row change ready for driver subscription consumers."""

    event_id: str
    row_id: str
    op: str
    text_repr: str = ""
    vector: Optional[list[float]] = None
    table_name: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "event_id": self.event_id,
            "row_id": self.row_id,
            "op": self.op,
            "text_repr": self.text_repr,
            "table_name": self.table_name,
            "ts": self.timestamp,
        }
        if self.vector is not None:
            d["vector"] = self.vector
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RowEventRecord":
        return cls(
            event_id=str(d.get("event_id", "")),
            row_id=str(d["row_id"]),
            op=str(d.get("op", d.get("event_type", "INSERT"))).upper(),
            text_repr=str(d.get("text_repr", "")),
            vector=list(d["vector"]) if d.get("vector") is not None else None,
            table_name=str(d.get("table_name", "")),
            timestamp=float(d.get("ts", time.time())),
        )


class RowEventHub:
    """
    Thread-safe fan-out hub for WAL row events.

    Maintains a replay buffer for new subscribers and pushes live events
    to all active asyncio subscriber queues.
    """

    def __init__(self, replay_buffer: int = 50_000) -> None:
        self._lock = threading.RLock()
        self._history: list[RowEventRecord] = []
        self._replay_buffer = replay_buffer
        self._subscribers: list[asyncio.Queue[RowEventRecord]] = []
        self.events_published: int = 0
        self.events_dropped: int = 0

    def publish(self, record: RowEventRecord) -> None:
        with self._lock:
            self._history.append(record)
            if len(self._history) > self._replay_buffer:
                overflow = len(self._history) - self._replay_buffer
                self._history = self._history[overflow:]
            self.events_published += 1
            dead: list[asyncio.Queue[RowEventRecord]] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(record)
                except asyncio.QueueFull:
                    self.events_dropped += 1
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    def snapshot(self) -> list[RowEventRecord]:
        with self._lock:
            return list(self._history)

    def subscribe(self, *, replay: bool = True, queue_size: int = 10_000) -> asyncio.Queue[RowEventRecord]:
        q: asyncio.Queue[RowEventRecord] = asyncio.Queue(maxsize=queue_size)
        with self._lock:
            self._subscribers.append(q)
            if replay:
                for rec in self._history:
                    try:
                        q.put_nowait(rec)
                    except asyncio.QueueFull:
                        break
        return q

    def unsubscribe(self, q: asyncio.Queue[RowEventRecord]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    async def stream(self, *, replay: bool = True) -> AsyncIterator[RowEventRecord]:
        q = self.subscribe(replay=replay)
        try:
            while True:
                yield await q.get()
        finally:
            self.unsubscribe(q)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "history_size": len(self._history),
                "subscribers": len(self._subscribers),
                "events_published": self.events_published,
                "events_dropped": self.events_dropped,
            }


def ndjson_line(record: RowEventRecord) -> str:
    return json.dumps(record.to_dict()) + "\n"
