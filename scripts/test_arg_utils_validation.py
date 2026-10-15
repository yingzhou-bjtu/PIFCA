#!/usr/bin/env python3
"""Standalone checks for shared CLI argument parsers."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = REPO_ROOT / "system"
if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))

from utils.arg_utils import parse_bool


def expect_error(value) -> None:
    try:
        parse_bool(value)
    except argparse.ArgumentTypeError:
        return
    raise AssertionError(f"expected {value!r} to be rejected")


def main() -> int:
    for value in ("true", "TRUE", "1", "yes", "on", " true "):
        assert parse_bool(value) is True, value
    for value in ("false", "FALSE", "0", "no", "off", " false "):
        assert parse_bool(value) is False, value
    assert parse_bool(True) is True
    assert parse_bool(False) is False

    for value in ("", " maybe ", "2", "-1", None, 1):
        expect_error(value)

    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", type=parse_bool)
    assert parser.parse_args(["--flag", "false"]).flag is False
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(["--flag", "not-a-bool"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse accepted an invalid boolean value")

    print("ARG_UTILS_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
