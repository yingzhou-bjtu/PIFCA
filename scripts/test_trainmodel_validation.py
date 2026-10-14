#!/usr/bin/env python3
"""Validate lightweight train-model input boundaries."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = ROOT / "system"
if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))

from flcore.trainmodel.mobilenet_v2 import InvertedResidual


def main() -> int:
    try:
        InvertedResidual(inp=8, oup=8, stride=3, expand_ratio=1)
    except ValueError as exc:
        if "stride" not in str(exc):
            raise AssertionError(f"unexpected ValueError message: {exc}") from exc
    else:
        raise AssertionError("InvertedResidual accepted an unsupported stride")

    print("PIFCA_TRAINMODEL_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
