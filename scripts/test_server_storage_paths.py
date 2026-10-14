#!/usr/bin/env python3
"""Validate configurable PIFCA server storage paths."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import torch


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = ROOT / "system"
if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))

from flcore.servers.serverbase import Server
from flcore.servers.serverpFedMe import pFedMe


def bare_server(server_class=Server):
    return object.__new__(server_class)


def configure_storage_fields(server, root: Path) -> None:
    server.model_dir = str(root / "checkpoints")
    server.results_dir = str(root / "metrics")
    server.dataset = "demo_dataset"
    server.algorithm = "DemoAlgorithm"
    server.goal = "demo_goal"
    server.times = 3


def main() -> int:
    with TemporaryDirectory(prefix="pifca-server-storage-") as temporary:
        root = Path(temporary)

        server = bare_server()
        configure_storage_fields(server, root)
        server.global_model = torch.nn.Linear(2, 1)
        server.save_global_model()
        checkpoint_path = root / "checkpoints" / "demo_dataset" / "DemoAlgorithm_server.pt"
        if not checkpoint_path.is_file():
            raise AssertionError(f"checkpoint was not saved at {checkpoint_path}")
        if not server.model_exists():
            raise AssertionError("model_exists did not use the configured model_dir")

        server.rs_test_acc = [0.25, 0.5]
        server.rs_test_auc = [0.3, 0.6]
        server.rs_train_loss = [1.0, 0.5]
        server.save_results()
        result_path = root / "metrics" / "demo_dataset_DemoAlgorithm_demo_goal_3.h5"
        if not result_path.is_file():
            raise AssertionError(f"result file was not saved at {result_path}")
        with h5py.File(result_path, "r") as handle:
            if list(handle["rs_test_acc"]) != [0.25, 0.5]:
                raise AssertionError("base server result file used unexpected content")

        pfedme_server = bare_server(pFedMe)
        configure_storage_fields(pfedme_server, root)
        pfedme_server.rs_test_acc_per = [0.75]
        pfedme_server.rs_train_acc_per = [0.5]
        pfedme_server.rs_train_loss_per = [0.25]
        pfedme_server.save_results()
        with h5py.File(result_path, "r") as handle:
            if list(handle["rs_train_acc"]) != [0.5]:
                raise AssertionError("pFedMe result file did not use configured results_dir")

    print("PIFCA_SERVER_STORAGE_PATHS_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
