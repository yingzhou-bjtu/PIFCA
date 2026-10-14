#!/usr/bin/env python3
"""Validate lightweight language utility input boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from utils.language_utils import letter_to_vec, val_to_vec


def expect_error(error_type: type[Exception], message: str, size: object, val: object) -> None:
    try:
        val_to_vec(size, val)
    except error_type as exc:
        if message not in str(exc):
            raise AssertionError(f"unexpected error message: {exc}") from exc
    else:
        raise AssertionError(f"val_to_vec accepted invalid input: size={size!r}, val={val!r}")


def main() -> int:
    if val_to_vec(4, 2) != [0, 0, 1, 0]:
        raise AssertionError("val_to_vec returned the wrong one-hot vector")
    if val_to_vec(np.int64(3), np.int64(1)) != [0, 1, 0]:
        raise AssertionError("val_to_vec rejected numpy integer inputs")

    expect_error(TypeError, "size must be an integer", 4.0, 1)
    expect_error(ValueError, "size must be positive", 0, 0)
    expect_error(TypeError, "val must be an integer", 4, 1.5)
    expect_error(ValueError, "val must be in range", 4, 4)

    if sum(letter_to_vec("A")) != 1:
        raise AssertionError("letter_to_vec did not return a one-hot vector")

    print("PIFCA_LANGUAGE_UTILS_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
