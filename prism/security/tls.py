"""
prism.security.tls — gRPC TLS credential helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_server_credentials(
    cert_path: str | Path,
    key_path: str | Path,
    *,
    require_client_cert: bool = False,
    ca_path: Optional[str | Path] = None,
) -> object:
    """Load grpc.ssl_server_credentials from PEM files."""
    import grpc

    cert_path = Path(cert_path)
    key_path = Path(key_path)
    with cert_path.open("rb") as cf, key_path.open("rb") as kf:
        pair = [(kf.read(), cf.read())]
    root_certs = None
    if ca_path:
        root_certs = Path(ca_path).read_bytes()
    if require_client_cert and root_certs:
        return grpc.ssl_server_credentials(pair, root_certs, require_client_certificate=True)
    return grpc.ssl_server_credentials(pair)


def load_client_credentials(
    cert_path: str | Path,
    *,
    key_path: Optional[str | Path] = None,
    ca_path: Optional[str | Path] = None,
) -> object:
    """Load grpc.ssl_channel_credentials for mutual TLS clients."""
    import grpc

    root = Path(ca_path).read_bytes() if ca_path else None
    with Path(cert_path).open("rb") as cf:
        cert = cf.read()
    key = Path(key_path).read_bytes() if key_path else None
    if key:
        return grpc.ssl_channel_credentials(root, ((key, cert),))
    return grpc.ssl_channel_credentials(root or cert)
