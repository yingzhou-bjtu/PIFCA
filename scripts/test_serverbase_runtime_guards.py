#!/usr/bin/env python3
"""Validate PIFCA server runtime state guards."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = ROOT / "system"
if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))

from flcore.servers.serverbase import Server


def expect_error(error_type: type[Exception], message: str, action) -> None:
    try:
        action()
    except error_type as exc:
        if message not in str(exc):
            raise AssertionError(f"unexpected error message: {exc}") from exc
    else:
        raise AssertionError(f"expected {error_type.__name__}: {message}")


def bare_server() -> Server:
    return object.__new__(Server)


def main() -> int:
    server = bare_server()
    server.clients = []
    expect_error(RuntimeError, "clients", server.send_models)
    expect_error(RuntimeError, "clients", lambda: server.send_attenmodels([]))

    server = bare_server()
    server.selected_clients = []
    expect_error(RuntimeError, "selected_clients", server.receive_models)
    expect_error(RuntimeError, "selected_clients", server.receive_clu_models)
    expect_error(RuntimeError, "selected_clients", server.clu_models)

    server = bare_server()
    server.uploaded_models = []
    expect_error(RuntimeError, "uploaded_models", server.aggregate_parameters)

    server = bare_server()
    server.dataset = "missing_dataset"
    server.algorithm = "missing_algorithm"
    expected_model_path = os.path.join("models", server.dataset, f"{server.algorithm}_server.pt")
    expect_error(FileNotFoundError, expected_model_path, server.load_model)

    print("PIFCA_SERVERBASE_GUARDS_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
