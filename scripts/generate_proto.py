#!/usr/bin/env python3
"""Regenerate gRPC Python stubs from proto/chorus.proto."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTO = ROOT / "proto" / "chorus.proto"
OUT = ROOT / "prism" / "wrapper" / "proto"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{ROOT / 'proto'}",
        f"--python_out={OUT}",
        f"--grpc_python_out={OUT}",
        str(PROTO),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    grpc_py = OUT / "chorus_pb2_grpc.py"
    text = grpc_py.read_text(encoding="utf-8")
    text = text.replace("import chorus_pb2 as chorus__pb2", "from . import chorus_pb2 as chorus__pb2")
    grpc_py.write_text(text, encoding="utf-8")
    print("Fixed relative import in chorus_pb2_grpc.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
