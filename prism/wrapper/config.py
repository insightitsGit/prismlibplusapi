"""
prism.wrapper.config — WrapperDaemon configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


def _env_bool(val: str) -> bool:
    return val.lower() in ("1", "true", "yes")


class DatabaseFlavor(str, Enum):
    POSTGRESQL  = "postgresql"
    MYSQL       = "mysql"
    COCKROACHDB = "cockroachdb"
    TIDB        = "tidb"


@dataclass
class WrapperConfig:
    """
    Full configuration for the Server Wrapper daemon.

    All fields can be set via environment variables with the PRISM_WRAPPER_
    prefix (e.g. PRISM_WRAPPER_DB_DSN) which takes precedence over values
    in a TOML config file.
    """

    # ── Database connection ──────────────────────────────────────────────
    db_flavor: DatabaseFlavor = DatabaseFlavor.POSTGRESQL
    db_dsn: str = ""               # e.g. "postgresql://user:pass@localhost/mydb"
    db_slot_name: str = "prism_slot"   # PostgreSQL replication slot name
    db_tables: list[str] = field(default_factory=list)  # empty = all tables

    # ── CHORUS Fabric gRPC server ────────────────────────────────────────
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    subscribe_http_port: int = 8081
    tenant_id: str = ""
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    tls_ca_path: Optional[str] = None
    require_client_cert: bool = False
    allow_insecure: bool = False

    # ── Vectorisation ────────────────────────────────────────────────────
    target_dim: int = 64            # JL projection output dimension
    key_ttl_seconds: float = 300.0  # cipher key rotation interval

    # ── Operational ─────────────────────────────────────────────────────
    pid_file: str = "/var/run/prism-wrapper.pid"
    log_level: str = "INFO"
    max_queue_size: int = 10_000    # in-memory event queue depth
    publish_batch_size: int = 64    # vectors per CHORUS frame
    reconnect_delay_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> "WrapperConfig":
        """Build config from PRISM_WRAPPER_* environment variables."""
        cfg = cls()
        mapping = {
            "PRISM_WRAPPER_DB_FLAVOR":   ("db_flavor",   DatabaseFlavor),
            "PRISM_WRAPPER_DB_DSN":      ("db_dsn",      str),
            "PRISM_WRAPPER_DB_SLOT":     ("db_slot_name", str),
            "PRISM_WRAPPER_GRPC_HOST":   ("grpc_host",   str),
            "PRISM_WRAPPER_GRPC_PORT":   ("grpc_port",   int),
            "PRISM_WRAPPER_SUBSCRIBE_PORT": ("subscribe_http_port", int),
            "PRISM_WRAPPER_TENANT_ID":   ("tenant_id",   str),
            "PRISM_WRAPPER_TLS_CERT":    ("tls_cert_path", str),
            "PRISM_WRAPPER_TLS_KEY":     ("tls_key_path",  str),
            "PRISM_WRAPPER_TLS_CA":      ("tls_ca_path",   str),
            "PRISM_WRAPPER_REQUIRE_CLIENT_CERT": ("require_client_cert", _env_bool),
            "PRISM_WRAPPER_ALLOW_INSECURE": ("allow_insecure", _env_bool),
            "PRISM_WRAPPER_TARGET_DIM":  ("target_dim",  int),
            "PRISM_WRAPPER_LOG_LEVEL":   ("log_level",   str),
        }
        for env_key, (attr, cast) in mapping.items():
            val = os.environ.get(env_key)
            if val is not None:
                setattr(cfg, attr, cast(val))
        tables_env = os.environ.get("PRISM_WRAPPER_DB_TABLES")
        if tables_env:
            cfg.db_tables = [t.strip() for t in tables_env.split(",") if t.strip()]
        return cfg

    @classmethod
    def from_toml(cls, path: str) -> "WrapperConfig":
        """Load config from a TOML file, then override with env vars."""
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore[no-reattr]

        with open(path, "rb") as f:
            data = tomllib.load(f)

        cfg = cls()
        db = data.get("database", {})
        grpc = data.get("grpc", {})
        vec = data.get("vectorisation", {})
        ops = data.get("operational", {})

        cfg.db_flavor      = DatabaseFlavor(db.get("flavor", cfg.db_flavor))
        cfg.db_dsn         = db.get("dsn", cfg.db_dsn)
        cfg.db_slot_name   = db.get("slot_name", cfg.db_slot_name)
        cfg.db_tables      = db.get("tables", cfg.db_tables)
        cfg.grpc_host      = grpc.get("host", cfg.grpc_host)
        cfg.grpc_port      = grpc.get("port", cfg.grpc_port)
        cfg.tls_cert_path  = grpc.get("tls_cert", cfg.tls_cert_path)
        cfg.tls_key_path   = grpc.get("tls_key", cfg.tls_key_path)
        cfg.tls_ca_path    = grpc.get("tls_ca", cfg.tls_ca_path)
        cfg.require_client_cert = grpc.get("require_client_cert", cfg.require_client_cert)
        cfg.target_dim     = vec.get("target_dim", cfg.target_dim)
        cfg.key_ttl_seconds = vec.get("key_ttl_seconds", cfg.key_ttl_seconds)
        cfg.log_level      = ops.get("log_level", cfg.log_level)
        cfg.pid_file       = ops.get("pid_file", cfg.pid_file)
        cfg.max_queue_size = ops.get("max_queue_size", cfg.max_queue_size)

        # env overrides TOML
        env = cls.from_env()
        for attr in vars(cfg):
            env_val = getattr(env, attr)
            default_val = getattr(cls(), attr)
            if env_val != default_val:
                setattr(cfg, attr, env_val)

        return cfg
