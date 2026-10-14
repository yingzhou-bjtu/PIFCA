#!/usr/bin/env python3
"""Validate lightweight memory reporter input boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = ROOT / "system"
if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))

from utils.mem_utils import MemReporter


def main() -> int:
    try:
        MemReporter(model=object())
    except TypeError as exc:
        if "torch.nn.Module" not in str(exc):
            raise AssertionError(f"unexpected TypeError message: {exc}") from exc
    else:
        raise AssertionError("MemReporter accepted a non-module model")

    model = torch.nn.Linear(2, 1)
    reporter = MemReporter(model=model)
    if not reporter.tensor_name:
        raise AssertionError("MemReporter did not register model parameters")

    print("PIFCA_MEM_UTILS_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
