#!/usr/bin/env python3
"""Validate PIFCA client runtime state guards."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = ROOT / "system"
if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))

from flcore.clients.clientapple import clientAPPLE
from flcore.clients.clientpcl import clientPCL


def expect_error(error_type: type[Exception], message: str, action) -> None:
    try:
        action()
    except error_type as exc:
        if message not in str(exc):
            raise AssertionError(f"unexpected error message: {exc}") from exc
    else:
        raise AssertionError(f"expected {error_type.__name__}: {message}")


def check_clientpcl_requires_aggregated_global_protos() -> None:
    client = object.__new__(clientPCL)
    client.protos = {"class-0": torch.zeros(2)}
    client.global_protos = {"class-0": []}
    client.local_epochs = 1
    client.train_slow = False
    client.model = nn.Linear(2, 2)
    client.load_train_data = lambda: []

    expect_error(
        RuntimeError,
        "global_protos",
        client.train,
    )


def check_clientapple_requires_models() -> None:
    client = object.__new__(clientAPPLE)
    client.model_cs = []
    client.model = nn.Linear(2, 2)

    expect_error(
        RuntimeError,
        "model_cs",
        client.aggregate_parameters,
    )


def main() -> int:
    check_clientpcl_requires_aggregated_global_protos()
    check_clientapple_requires_models()
    print("PIFCA_CLIENT_GUARDS_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
