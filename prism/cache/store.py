"""
prism.cache.store — Response Persistence Layer
===============================================

Stores and retrieves cached LLM responses keyed by wave-packet ID.

Two backends ship out of the box:

    InMemoryStore   — Default. Fast, zero setup, lost on process restart.
                      Best for: single-server deployments, development.

    SQLiteStore     — Persistent. Survives restarts. No extra services.
                      Best for: single-server production deployments.
                      File is created automatically on first use.

The store holds the raw LLM response object (dict, string, or any
JSON-serialisable value) alongside its TTL expiry timestamp.
Wave-packet IDs (UUIDs from PrismResonance) are the keys.

Thread safety: both implementations use a threading.RLock internally.
"""

from __future__ import annotations

import abc
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StoreError(Exception):
    """Base error for cache store operations."""


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """
    A single cached response.

    Attributes
    ----------
    packet_id:      The PrismResonance WavePacket ID this entry maps to.
    query_text:     Original query string (for debugging / analytics).
    response:       The raw LLM response — any JSON-serialisable value.
    created_at:     Unix timestamp of cache write.
    expires_at:     Unix timestamp after which this entry is invalid.
    hit_count:      Number of times this entry has been served from cache.
    tokens_saved:   Estimated tokens saved by cache hits (populated by caller).
    model:          LLM model name that produced this response.
    """

    packet_id: str
    query_text: str
    response: Any
    created_at: float
    expires_at: float
    hit_count: int = 0
    tokens_saved: int = 0
    model: str = ""

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "packet_id": self.packet_id,
            "query_text": self.query_text,
            "response": self.response,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hit_count": self.hit_count,
            "tokens_saved": self.tokens_saved,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        return cls(
            packet_id=d["packet_id"],
            query_text=d["query_text"],
            response=d["response"],
            created_at=d["created_at"],
            expires_at=d["expires_at"],
            hit_count=d.get("hit_count", 0),
            tokens_saved=d.get("tokens_saved", 0),
            model=d.get("model", ""),
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class CacheStore(abc.ABC):
    """Abstract contract for response persistence backends."""

    @abc.abstractmethod
    def save(self, entry: CacheEntry) -> None:
        """Persist a cache entry. Overwrites if packet_id already exists."""

    @abc.abstractmethod
    def load(self, packet_id: str) -> Optional[CacheEntry]:
        """
        Load a cache entry by packet_id.

        Returns None if not found or expired.
        Increments hit_count atomically on successful load.
        """

    @abc.abstractmethod
    def delete(self, packet_id: str) -> None:
        """Remove an entry by packet_id."""

    @abc.abstractmethod
    def purge_expired(self) -> int:
        """Remove all expired entries. Returns count of entries removed."""

    @abc.abstractmethod
    def count(self) -> int:
        """Return total number of (including expired) entries."""

    @abc.abstractmethod
    def total_hits(self) -> int:
        """Return the sum of hit_count across all entries."""

    @abc.abstractmethod
    def total_tokens_saved(self) -> int:
        """Return the sum of tokens_saved across all entries."""


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------


class InMemoryStore(CacheStore):
    """
    In-process dict-backed store.

    Entries are lost on process restart. For single-server deployments
    where the wave cache (PrismResonance) is also in-process, this is
    fine — both will be warm after the first few requests.

    Thread safety: all mutations hold self._lock.
    """

    def __init__(self, max_size: int = 50_000) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._max_size = max_size

    def save(self, entry: CacheEntry) -> None:
        with self._lock:
            if len(self._store) >= self._max_size:
                self._evict_oldest()
            self._store[entry.packet_id] = entry

    def load(self, packet_id: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._store.get(packet_id)
            if entry is None:
                return None
            if entry.is_expired():
                del self._store[packet_id]
                return None
            entry.hit_count += 1
            return entry

    def delete(self, packet_id: str) -> None:
        with self._lock:
            self._store.pop(packet_id, None)

    def purge_expired(self) -> int:
        with self._lock:
            expired = [pid for pid, e in self._store.items() if e.is_expired()]
            for pid in expired:
                del self._store[pid]
        return len(expired)

    def count(self) -> int:
        with self._lock:
            return len(self._store)

    def total_hits(self) -> int:
        with self._lock:
            return sum(e.hit_count for e in self._store.values())

    def total_tokens_saved(self) -> int:
        with self._lock:
            return sum(e.tokens_saved for e in self._store.values())

    def _evict_oldest(self) -> None:
        """Evict the 10% oldest entries when the store is full."""
        n_evict = max(1, self._max_size // 10)
        by_age = sorted(self._store.items(), key=lambda kv: kv[1].created_at)
        for pid, _ in by_age[:n_evict]:
            del self._store[pid]


# ---------------------------------------------------------------------------
# SQLiteStore
# ---------------------------------------------------------------------------


class SQLiteStore(CacheStore):
    """
    SQLite-backed persistent store.

    The database file is created automatically on first use.
    Uses WAL mode for concurrent reads without blocking writes.

    Best for: single-server production deployments that need
    persistence across restarts without running extra services.

    For multi-server deployments, each app node maintains its own
    SQLite file — the PrismResonance wave cache is the source of
    truth for similarity lookups; SQLite is only for response storage.

    Usage:
        store = SQLiteStore("/var/lib/prismcache/responses.db")
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS cache_entries (
        packet_id    TEXT PRIMARY KEY,
        query_text   TEXT NOT NULL,
        response_json TEXT NOT NULL,
        created_at   REAL NOT NULL,
        expires_at   REAL NOT NULL,
        hit_count    INTEGER NOT NULL DEFAULT 0,
        tokens_saved INTEGER NOT NULL DEFAULT 0,
        model        TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries (expires_at);
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = self._connect()
        logger.info("SQLiteStore: opened at '%s'.", db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,   # we manage thread safety with _lock
            timeout=10,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(self._SCHEMA)
        conn.commit()
        return conn

    def save(self, entry: CacheEntry) -> None:
        response_json = json.dumps(entry.response, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cache_entries
                    (packet_id, query_text, response_json, created_at,
                     expires_at, hit_count, tokens_saved, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(packet_id) DO UPDATE SET
                    response_json = excluded.response_json,
                    expires_at    = excluded.expires_at,
                    hit_count     = excluded.hit_count,
                    tokens_saved  = excluded.tokens_saved
                """,
                (
                    entry.packet_id,
                    entry.query_text,
                    response_json,
                    entry.created_at,
                    entry.expires_at,
                    entry.hit_count,
                    entry.tokens_saved,
                    entry.model,
                ),
            )
            self._conn.commit()

    def load(self, packet_id: str) -> Optional[CacheEntry]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM cache_entries WHERE packet_id = ?",
                (packet_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None

            cols = [d[0] for d in cur.description]
            data = dict(zip(cols, row))

            if time.time() > data["expires_at"]:
                self._conn.execute(
                    "DELETE FROM cache_entries WHERE packet_id = ?", (packet_id,)
                )
                self._conn.commit()
                return None

            # Increment hit count
            self._conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE packet_id = ?",
                (packet_id,),
            )
            self._conn.commit()

        return CacheEntry(
            packet_id=data["packet_id"],
            query_text=data["query_text"],
            response=json.loads(data["response_json"]),
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            hit_count=data["hit_count"] + 1,
            tokens_saved=data["tokens_saved"],
            model=data["model"],
        )

    def delete(self, packet_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM cache_entries WHERE packet_id = ?", (packet_id,)
            )
            self._conn.commit()

    def purge_expired(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM cache_entries WHERE expires_at < ?", (time.time(),)
            )
            self._conn.commit()
        return cur.rowcount

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM cache_entries")
            return int(cur.fetchone()[0])

    def total_hits(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COALESCE(SUM(hit_count), 0) FROM cache_entries")
            return int(cur.fetchone()[0])

    def total_tokens_saved(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COALESCE(SUM(tokens_saved), 0) FROM cache_entries")
            return int(cur.fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
