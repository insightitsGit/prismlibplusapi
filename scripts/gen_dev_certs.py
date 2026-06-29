#!/usr/bin/env python3
"""
Generate self-signed CA + server + client certs for local mTLS development.

Requires OpenSSL on PATH.

Usage:
    python scripts/gen_dev_certs.py
    python scripts/gen_dev_certs.py --out certs/dev

Outputs:
    ca.crt, ca.key          — Certificate authority
    server.crt, server.key  — Wrapper gRPC server
    client.crt, client.key  — PrismDriver client

Then:
    export PRISM_WRAPPER_TLS_CERT=certs/dev/server.crt
    export PRISM_WRAPPER_TLS_KEY=certs/dev/server.key
    export PRISM_WRAPPER_TLS_CA=certs/dev/ca.crt
    export PRISM_WRAPPER_REQUIRE_CLIENT_CERT=true

    export PRISM_DRIVER_TLS_CA=certs/dev/ca.crt
    export PRISM_DRIVER_TLS_CLIENT_CERT=certs/dev/client.crt
    export PRISM_DRIVER_TLS_CLIENT_KEY=certs/dev/client.key
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main() -> int:
    if not shutil.which("openssl"):
        print("ERROR: openssl not found on PATH. Install OpenSSL or use WSL.", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description="Generate dev mTLS certificates")
    parser.add_argument("--out", default="certs/dev", help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ca_key = out / "ca.key"
    ca_crt = out / "ca.crt"
    srv_key = out / "server.key"
    srv_csr = out / "server.csr"
    srv_crt = out / "server.crt"
    cli_key = out / "client.key"
    cli_csr = out / "client.csr"
    cli_crt = out / "client.crt"

    subj_ca = "/CN=PrismDevCA"
    subj_srv = "/CN=prism-wrapper.local"
    subj_cli = "/CN=prism-driver"

    _run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
          "-keyout", str(ca_key), "-out", str(ca_crt), "-days", "365", "-subj", subj_ca], out)

    _run(["openssl", "req", "-newkey", "rsa:2048", "-nodes",
          "-keyout", str(srv_key), "-out", str(srv_csr), "-subj", subj_srv], out)
    _run(["openssl", "x509", "-req", "-in", str(srv_csr), "-CA", str(ca_crt), "-CAkey", str(ca_key),
          "-CAcreateserial", "-out", str(srv_crt), "-days", "365"], out)

    _run(["openssl", "req", "-newkey", "rsa:2048", "-nodes",
          "-keyout", str(cli_key), "-out", str(cli_csr), "-subj", subj_cli], out)
    _run(["openssl", "x509", "-req", "-in", str(cli_csr), "-CA", str(ca_crt), "-CAkey", str(ca_key),
          "-CAcreateserial", "-out", str(cli_crt), "-days", "365"], out)

    for f in (srv_csr, cli_csr, out / "ca.srl"):
        f.unlink(missing_ok=True)

    print(f"\nCertificates written to {out.resolve()}")
    print("\nWrapper (DB node):")
    print(f"  PRISM_WRAPPER_TLS_CERT={srv_crt}")
    print(f"  PRISM_WRAPPER_TLS_KEY={srv_key}")
    print(f"  PRISM_WRAPPER_TLS_CA={ca_crt}")
    print("  PRISM_WRAPPER_REQUIRE_CLIENT_CERT=true")
    print("\nDriver (app node):")
    print(f"  PRISM_DRIVER_TLS_CA={ca_crt}")
    print(f"  PRISM_DRIVER_TLS_CLIENT_CERT={cli_crt}")
    print(f"  PRISM_DRIVER_TLS_CLIENT_KEY={cli_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
