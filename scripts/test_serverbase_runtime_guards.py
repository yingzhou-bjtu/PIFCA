#!/usr/bin/env python3
"""Validate PIFCA server runtime state guards."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace


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


def bare_subclass(module_name: str, class_name: str):
    module = import_module(f"flcore.servers.{module_name}")
    server_class = getattr(module, class_name)
    return object.__new__(server_class)


def check_subclass_collection_guard(
    module_name: str,
    class_name: str,
    field_name: str,
    method_name: str,
    *method_args,
) -> None:
    server = bare_subclass(module_name, class_name)
    setattr(server, field_name, [])
    expect_error(RuntimeError, field_name, lambda: getattr(server, method_name)(*method_args))


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
    server.model_dir = "models"
    server.dataset = "missing_dataset"
    server.algorithm = "missing_algorithm"
    expected_model_path = os.path.join("models", server.dataset, f"{server.algorithm}_server.pt")
    expect_error(FileNotFoundError, expected_model_path, server.load_model)

    collection_cases = (
        ("serverdyn", "FedDyn", "uploaded_models", "aggregate_parameters"),
        ("serverdyn", "FedDyn", "uploaded_models", "update_server_state"),
        ("serverdistill", "FedDistill", "clients", "send_logits"),
        ("serverdistill", "FedDistill", "selected_clients", "receive_logits"),
        ("servergen", "FedGen", "clients", "send_models"),
        ("servergen", "FedGen", "selected_clients", "receive_models"),
        ("serverpac", "FedPAC", "clients", "send_protos"),
        ("serverpac", "FedPAC", "selected_clients", "receive_protos"),
        ("serverpac", "FedPAC", "selected_clients", "receive_models"),
        ("serverda", "PFL_DA", "selected_clients", "receive_models"),
        ("servercp", "FedCP", "clients", "send_models"),
        ("servercp", "FedCP", "uploaded_models", "aggregate_parameters"),
        ("servercp", "FedCP", "selected_clients", "receive_models"),
        ("servergpfl", "GPFL", "selected_clients", "receive_models"),
        ("serverapple", "APPLE", "clients", "send_models"),
        ("serverlg", "LG_FedAvg", "selected_clients", "receive_models"),
        ("serverpcl", "FedPCL", "clients", "send_protos"),
        ("serverpcl", "FedPCL", "selected_clients", "receive_protos"),
        ("servergc", "FedGC", "clients", "send_models"),
        ("servergc", "FedGC", "selected_clients", "receive_models"),
        ("serverkd", "FedKD", "clients", "send_models"),
        ("serverkd", "FedKD", "selected_clients", "receive_models"),
        ("serverkd", "FedKD", "uploaded_models", "aggregate_parameters"),
        ("serverper", "FedPer", "selected_clients", "receive_models"),
        ("servergh", "FedGH", "clients", "send_models"),
        ("servergh", "FedGH", "selected_clients", "receive_protos"),
        ("serverfml", "FML", "selected_clients", "receive_models"),
        ("serverfml", "FML", "uploaded_models", "aggregate_parameters"),
        ("serverscaffold", "SCAFFOLD", "clients", "send_models"),
        ("serverscaffold", "SCAFFOLD", "selected_clients", "receive_models"),
        ("serverproto", "FedProto", "clients", "send_protos"),
        ("serverproto", "FedProto", "selected_clients", "receive_protos"),
        ("serverfomo", "FedFomo", "selected_clients", "send_models"),
        ("serverfomo", "FedFomo", "selected_clients", "receive_models"),
        ("serverala", "FedALA", "clients", "send_models"),
        ("serverbabu", "FedBABU", "selected_clients", "receive_models"),
        ("serveramp", "FedAMP", "selected_clients", "send_models"),
        ("serverphp", "FedPHP", "selected_clients", "send_models", None),
        ("serverala_clu", "FedALA_clu", "clients", "send_models"),
        ("serverrep", "FedRep", "selected_clients", "receive_models"),
    )
    for case in collection_cases:
        check_subclass_collection_guard(*case)

    server = bare_subclass("servercac", "FedCAC")
    server.args = SimpleNamespace(beta=0)
    expect_error(ValueError, "args.beta", server.get_customized_global_models)

    print("PIFCA_SERVER_GUARDS_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
