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

from utils.arg_utils import parse_bool, validate_experiment_args


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

    valid = argparse.Namespace(
        num_clients=10,
        batch_size=10,
        global_rounds=100,
        local_epochs=1,
        times=1,
        eval_gap=1,
        dlg_gap=100,
        batch_num_per_client=2,
        feature_dim=512,
        vocab_size=32000,
        max_len=200,
        K=5,
        M=5,
        itk=4000,
        plocal_epochs=1,
        fine_tuning_epochs=10,
        noise_dim=512,
        hidden_dim=512,
        server_epochs=1000,
        layer_idx=2,
        top_cnt=100,
        prev=0,
        num_new_clients=0,
        fine_tuning_epoch_new=0,
        rand_percent=80,
        local_learning_rate=0.001,
        learning_rate_decay_gamma=0.99,
        time_threthold=10000.0,
        p_learning_rate=0.01,
        alphaK=1.0,
        sigma=1.0,
        lamda=1.0,
        alpha=1.0,
        tau=1.0,
        L=1.0,
        generator_learning_rate=0.005,
        server_learning_rate=1.0,
        eta=1.0,
        mentee_learning_rate=0.005,
        T_start=0.95,
        T_end=0.98,
        beta=0.0,
        mu=0.0,
        dr_learning_rate=0.0,
        momentum=0.1,
        kl_weight=0.0,
        join_ratio=1.0,
        client_drop_rate=0.0,
        train_slow_rate=0.0,
        send_slow_rate=0.0,
    )
    assert validate_experiment_args(valid) is valid

    for field_name, value in (
        ("num_clients", 0),
        ("num_clients", True),
        ("local_learning_rate", float("nan")),
        ("join_ratio", 1.5),
        ("client_drop_rate", -0.1),
        ("rand_percent", 101),
    ):
        invalid = argparse.Namespace(**{**vars(valid), field_name: value})
        try:
            validate_experiment_args(invalid)
        except ValueError as exc:
            assert field_name in str(exc)
        else:
            raise AssertionError(f"invalid {field_name} was accepted: {value!r}")

    print("ARG_UTILS_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
